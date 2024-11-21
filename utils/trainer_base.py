from model import GraphEncoder, InstructGLM
from transformers import LlamaConfig, get_scheduler, LlamaForCausalLM
import torch
from .utils import *
from .instruction_preprocess import InstructionDataset
from peft import (    # LoRA Setting
    LoraConfig,
    get_peft_model,
)

class TrainerBase(object):
    def __init__(self, args, device):
        self.args = args
        self.cur_device = device


    def create_dataloader(self, tokenizer):
        DatasetCls = InstructionDataset
            
        if self.args.inference:
            test_dataset = DatasetCls(tokenizer, self.args, mode="test")
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, drop_last=False,
                                                    pin_memory=True, shuffle=False, collate_fn=test_dataset.collate_fn)
            return test_loader
        else:
            train_dataset = DatasetCls(tokenizer, self.args, mode="train")


            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, drop_last=True,
                                                    pin_memory=True, shuffle=True, collate_fn=train_dataset.collate_fn)

            return train_loader


    def create_config(self):
        if self.args.config_class == 'LlamaConfig':
            config_class = LlamaConfig

        config = config_class.from_pretrained(self.args.backbone)

        config.dropout_rate = self.args.dropout
        config.dropout = self.args.dropout
        config.attention_dropout = self.args.dropout
        config.activation_dropout = self.args.dropout

        return config

    def create_model(self):
        config = self.create_config()

        model = InstructGLM.from_pretrained(
            self.args.backbone,
            config=config,
            torch_dtype=torch.bfloat16,
            # use_cache=True, 
            # low_cpu_mem_usage=True,
            device_map={"": self.cur_device}
        )

        llama_embeds = model.get_input_embeddings().weight.data
        
        if self.args.freeze_llama:
            for n, p in model.named_parameters():
                p.requires_grad_(False)
        else:
            lora_r = 8
            lora_alpha = 16
            lora_target_modules=['q_proj','k_proj','v_proj', 'o_proj','lm_head']  # Select LoRA tuning modules.
            lora_dropout = 0.05
            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            model = get_peft_model(model, LORA_config)


        node_token=torch.zeros(110, llama_embeds.shape[1]).to(device=self.cur_device, dtype=llama_embeds.dtype)
        llama_embeds=torch.cat([llama_embeds, node_token],dim=0)

        self.args.gnn_output = llama_embeds.shape[1]
        first_model = GraphEncoder(self.args, llama_embed=llama_embeds).to(self.cur_device, dtype=torch.bfloat16)
        if not self.args.inference:
            first_model.GT.load_state_dict(torch.load(f'./saved_model/gnn/{self.args.pretrain_gnn}'))
        for n, p in first_model.named_parameters():
            if n.split('.')[0] == 'GT':
                p.requires_grad_(False)

        return first_model, model


    def create_optimizer_and_scheduler(self, first_model, model, train_loader):
        #no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in first_model.named_parameters() if p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.args.weight_decay,
            }
        ]

        optimizer_class = get_optimizer(self.args.optim)
        optimizer = optimizer_class(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            eps=self.args.adam_eps,
            betas=(self.args.adam_beta1, self.args.adam_beta1)) # betas=(0.9, 0.95)

        # 根据warmup配置，设置warmup
        total_steps = len(train_loader) * self.args.epoch
        num_warmup_steps = int(total_steps * self.args.warmup_ratio)
        assert num_warmup_steps <= total_steps, \
            'num_warmup_steps {} is too large, more than total_steps {}'.format(num_warmup_steps, total_steps)
        warmup_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=total_steps
            )

        return optimizer, warmup_scheduler
