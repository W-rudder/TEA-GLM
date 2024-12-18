import json
import time
import transformers
import torch
import os
import gc
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

from config import *
from model import *
from utils import *

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

from transformers import LlamaTokenizer


def main(args, SEED):
    run_time = time.strftime("%Y%m%d%H%M", time.localtime())
    wandb_name = f"{args.prefix}_EXP{SEED}_{run_time}"
    if args.inference:
        group = f"{args.dataset}_{args.test_dataset}"
    else:
        group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.backbone],
                                                "group": group,
                                                "name": wandb_name,
                                                "config": args}
                                           },
                              )

    seed_everything(seed=SEED)
    accelerator.print(args)

    with accelerator.main_process_first():

        tokenizer = LlamaTokenizer.from_pretrained(args.backbone)
        tokenizer.pad_token=tokenizer.unk_token
        special={'additional_special_tokens': ['<Node {}>'.format(i) for i in range(1, 110)]}   # Add a new special token as place holder
        tokenizer.add_special_tokens(special)

    accelerator.wait_for_everyone()

    cur_device = torch.cuda.current_device()
    trainer = TrainerBase(args, cur_device)

    accelerator.print('Building DataLoader')
    if not args.inference:
        train_loader = trainer.create_dataloader(tokenizer)
    else:
        test_loader = trainer.create_dataloader(tokenizer)
        train_loader = test_loader

    accelerator.print('Building Model')
    first_model, model = trainer.create_model()
    torch.set_default_tensor_type(torch.FloatTensor)

    accelerator.print('Building Optimizer')
    optimizer, warmup_scheduler = trainer.create_optimizer_and_scheduler(first_model, model, train_loader)

    trainable_params, all_param = print_trainable_params(first_model, model)
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    if not os.path.exists('./saved_model/first_model'):
        os.mkdir('./saved_model/first_model')

    first_model_path = './saved_model/first_model/{}_fm_{}_epoch{}_{}.pth'
    model_path = './saved_model/model/{}_m_{}_epoch{}_{}.pth'


    if not args.inference:
        first_model, model, train_loader, optimizer, warmup_scheduler = accelerator.prepare(first_model, model, train_loader,
                                                                                        optimizer, warmup_scheduler)
        accelerator.print('Training')
        num_training_steps = args.epoch * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(args.epoch):

            model.train()
            first_model.train()
            epoch_loss, accum_loss = 0., 0.

            for step, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()

                    input_ids = batch['input_ids']
                    is_node = batch['is_node']
                    labels = batch["target_ids"]
                    attention_mask = batch['attn_mask']
                    graph = batch['graph']
                    
                    embeds = first_model(
                        input_ids=input_ids,
                        is_node=is_node,
                        graph=graph
                    )
                    output=model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

                    loss = output['loss']
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                    accelerator.clip_grad_norm_(optimizer.param_groups[1]['params'], 0.1)

                    optimizer.step()
                    warmup_scheduler.step()

                    epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()


                if (step + 1) % args.grad_steps == 0:
                    graph_lr = optimizer.param_groups[0]["lr"]
                    lora_lr = optimizer.param_groups[1]["lr"]

                    accelerator.print({'Graph Lr': graph_lr, 'Lora Lr': lora_lr})
                    accelerator.print({'Accum Loss': accum_loss / args.grad_steps})
                    accelerator.log({'Graph Lr': graph_lr, 'Lora Lr': lora_lr})
                    accelerator.log({'Accum Loss': accum_loss / args.grad_steps})
                    accum_loss = 0.

                progress_bar.update(1)

            accelerator.print(f"Epoch: {epoch}|{args.epoch}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
            accelerator.log({
                'Epoch': epoch,
                'Loss': epoch_loss / len(train_loader)
                })

            # train only one epoch and save model
            if args.epoch == 1:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.save(accelerator.unwrap_model(first_model).state_dict(), first_model_path.format(args.prefix, args.dataset, epoch, 'end'))
                    if not args.freeze_llama:
                        accelerator.save(accelerator.unwrap_model(model).state_dict(), model_path.format(args.prefix, args.dataset, epoch, 'end'))
            best_epoch = args.best_epoch
     
        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        accelerator.wait_for_everyone()
    else:
        first_model, model, test_loader = accelerator.prepare(first_model, model, test_loader)
        best_epoch = args.best_epoch

        # Step 5. Evaluating
        accelerator.print('Evaluating')
        with accelerator.main_process_first():
            first_model = accelerator.unwrap_model(first_model)
            first_model.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, best_epoch, 'end')))
            # first_model.GT.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, best_epoch, 'end')))
            model = model.cuda() # transformers bug
            model = accelerator.unwrap_model(model)
            if not args.freeze_llama:
                model.load_state_dict(torch.load(model_path.format(args.prefix, args.dataset, best_epoch, 'end')))

        first_model.eval()
        model.eval()
        samples_seen = 0
        eval_output = []
        eval_label = []

        progress_bar_test = tqdm(range(len(test_loader)))
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                input_ids = batch['input_ids']
                is_node = batch['is_node']
                attention_mask = batch['attn_mask']
                graph = batch['graph']

                embeds = first_model(
                    input_ids=input_ids,
                    is_node=is_node,
                    graph=graph
                )

                results = model.g_step(in_embeds=embeds, attention_mask=attention_mask)
                results = accelerator.pad_across_processes(results, dim=1, pad_index=tokenizer.pad_token_id)
                results_gathered = accelerator.gather(results).cpu().numpy()

                labels = accelerator.pad_across_processes(
                    batch["target_ids"],
                    dim=1,
                    pad_index=tokenizer.pad_token_id)
                labels_gathered = accelerator.gather(labels).cpu().numpy()

                if accelerator.num_processes > 1:
                    if step == len(test_loader) - 1:
                        results_gathered = results_gathered[
                                                    : len(test_loader.dataset) - samples_seen]
                        labels_gathered = labels_gathered[
                                                    : len(test_loader.dataset) - samples_seen]
                    else:
                        samples_seen += len(results_gathered)
                labels_gathered = np.where(labels_gathered != -100, labels_gathered, tokenizer.pad_token_id)
                accelerator.print(tokenizer.batch_decode(results_gathered, skip_special_tokens=True))
                # accelerator.print(tokenizer.batch_decode(labels_gathered, skip_special_tokens=True))

                eval_output.append(results_gathered)
                eval_label.append(labels_gathered)
            progress_bar_test.update(1)

        # Step 6. Post-processing & Evaluating
        res_path = f'./results/{args.test_dataset}/{args.prefix}_model_results.txt'
        label_path = f'./results/{args.test_dataset}/{args.prefix}_model_labels.txt'

        if not os.path.exists(f'./results/{args.test_dataset}'):
            os.makedirs(f'./results/{args.test_dataset}')

        if accelerator.is_local_main_process:
            eval_pred, eval_decode_label = output_decode(eval_output, eval_label, tokenizer)
            with open(res_path, 'w') as f:
                json.dump(eval_pred, f)
            with open(label_path, 'w') as f:
                json.dump(eval_decode_label, f)
    

if __name__ == "__main__":

    args = parse_args()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
                                  gradient_accumulation_steps=args.grad_steps)
        if args.seed != -1:
            SEED = args.seed
        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()