import re
import numpy as np
import torch
import torch.distributed as dist
import collections
import logging
import random, os
import math
import pickle, json
import gzip
import gc
import dgl
from typing import Iterable, Optional, Dict
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def adjust_learning_rate(param_group, LR, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epoch - args.warmup_epochs))
        )
    param_group["lr"] = lr
    return lr

class LossMeter(object):   # Logging purpose
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def set_trainable_params_new(self):
    param_adapter, param_lora  = [],  []
    adapter = ["graph_adapter", "prefix_adapter", "up_projection", "down_projection"]

    for name, param in self.named_parameters():
        if any(n in name for n in adapter):
            param.requires_grad = True
            param.data = param.data.float()
            param_adapter.append(param)
        elif "lora" in name:
            param.requires_grad = True
            param.data = param.data.float()
            param_lora.append(param)
        else:
            param.requires_grad = False

    return param_adapter, param_lora


def print_trainable_params(first_model, model):
    trainable_params = 0
    all_param = 0

    for _, param in first_model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer

def output_decode(eval_output, eval_label, tokenizer):
    eval_decode_output = []
    eval_decode_label = []
    assert len(eval_output) == len(eval_label)
    for i in range(len(eval_output)):
        batch_output = eval_output[i]
        label_output = eval_label[i]
        eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=True))
        eval_decode_label.extend(tokenizer.batch_decode(label_output, skip_special_tokens=True))
    assert len(eval_decode_label) == len(eval_decode_output)

    # eval_decode_output = [item.split('. ')[-1] for item in eval_decode_output]
    # eval_decode_output = [item.split('</s>')[0] for item in eval_decode_output]
    # eval_decode_output = [item.split('<s> ')[-1] for item in eval_decode_output]

    # eval_decode_label = [item.split(' ')[-1] for item in eval_decode_label]

    return eval_decode_output, eval_decode_label


def compute_metric(eval_pred, eval_decode_label):
    y = [1 if label.lower() == 'yes' else 0 for label in eval_decode_label]
    x = [1 if pred.lower() == 'yes' else 0 for pred in eval_pred]

    acc = accuracy_score(y, x)
    r = recall_score(y, x)
    p = precision_score(y, x)
    f1 = f1_score(y, x)

    return acc, r, p, f1


def compute_multi_class_metric(args, eval_pred, eval_decode_label):
    if args.graph_unsup:
        label_list = [
            'yes',
            'no'
        ]
    else:
        if args.zero_shot:
            if args.test_dataset == 'pubmed':
                label_list = [
                    'Experimentally induced diabetes',
                    'Type 1 diabetes',
                    'Type 2 diabetes'
                ]
            elif args.test_dataset == 'cora':
                label_list = [
                    'artificial intelligence, agents', 
                    'artificial intelligence, data mining', 
                    'artificial intelligence, expert systems', 
                    'artificial intelligence, games and search', 
                    'artificial intelligence, knowledge representation', 
                    'artificial intelligence, machine learning, case-based', 
                    'artificial intelligence, machine learning, genetic algorithms', 
                    'artificial intelligence, machine learning, neural networks', 
                    'artificial intelligence, machine learning, probabilistic methods', 
                    'artificial intelligence, machine learning, reinforcement learning', 
                    'artificial intelligence, machine learning, rule learning', 
                    'artificial intelligence, machine learning, theory', 
                    'artificial intelligence, nlp', 
                    'artificial intelligence, planning', 
                    'artificial intelligence, robotics', 
                    'artificial intelligence, speech', 
                    'artificial intelligence, theorem proving', 
                    'artificial intelligence, vision and pattern recognition', 
                    'data structures, algorithms and theory, computational complexity', 
                    'data structures, algorithms and theory, computational geometry', 
                    'data structures, algorithms and theory, formal languages', 
                    'data structures, algorithms and theory, hashing', 
                    'data structures, algorithms and theory, logic', 
                    'data structures, algorithms and theory, parallel', 
                    'data structures, algorithms and theory, quantum computing', 
                    'data structures, algorithms and theory, randomized', 
                    'data structures, algorithms and theory, sorting', 
                    'databases, concurrency', 
                    'databases, deductive', 
                    'databases, object oriented', 
                    'databases, performance', 
                    'databases, query evaluation', 
                    'databases, relational', 
                    'databases, temporal', 
                    'encryption and compression, compression', 
                    'encryption and compression, encryption', 
                    'encryption and compression, security', 
                    'hardware and architecture, distributed architectures', 
                    'hardware and architecture, high performance computing', 
                    'hardware and architecture, input output and storage', 
                    'hardware and architecture, logic design', 
                    'hardware and architecture, memory structures', 
                    'hardware and architecture, microprogramming', 
                    'hardware and architecture, vlsi', 
                    'human computer interaction, cooperative', 
                    'human computer interaction, graphics and virtual reality', 
                    'human computer interaction, interface design', 
                    'human computer interaction, multimedia', 
                    'human computer interaction, wearable computers', 
                    'information retrieval, digital library', 
                    'information retrieval, extraction', 
                    'information retrieval, filtering', 
                    'information retrieval, retrieval', 
                    'nan', 
                    'networking, internet', 
                    'networking, protocols', 
                    'networking, routing', 
                    'networking, wireless', 
                    'operating systems, distributed', 
                    'operating systems, fault tolerance', 
                    'operating systems, memory management', 
                    'operating systems, realtime', 
                    'programming, compiler design', 
                    'programming, debugging', 
                    'programming, functional', 
                    'programming, garbage collection', 
                    'programming, java', 
                    'programming, logic', 
                    'programming, object oriented', 
                    'programming, semantics', 
                    'programming, software developmen']
            else:
                label_list = [
                    'cs.AI, Artificial Intelligence', 
                    'cs.CL, Computation and Language', 
                    'cs.CC, Computational Complexity', 
                    'cs.CE, Computational Engineering, Finance, and Science', 
                    'cs.CG, Computational Geometry', 
                    'cs.GT, Computer Science and Game Theory', 
                    'cs.CV, Computer Vision and Pattern Recognition', 
                    'cs.CY, Computers and Society',
                    'cs.CR, Cryptography and Security', 
                    'cs.DS, Data Structures and Algorithms', 
                    'cs.DB, Databases', 
                    'cs.DL, Digital Libraries', 
                    'cs.DM, Discrete Mathematics', 
                    'cs.DC, Distributed, Parallel, and Cluster Computing', 
                    'cs.ET, Emerging Technologies', 
                    'cs.FL, Formal Languages and Automata Theory', 
                    'cs.GL, General Literature', 
                    'cs.GR, Graphics', 
                    'cs.AR, Hardware Architecture', 
                    'cs.HC, Human-Computer Interaction', 
                    'cs.IR, Information Retrieval', 
                    'cs.IT, Information Theory', 
                    'cs.LO, Logic in Computer Science', 
                    'cs.LG, Machine Learning', 
                    'cs.MS, Mathematical Software', 
                    'cs.MA, Multiagent Systems', 
                    'cs.MM, Multimedia', 
                    'cs.NI, Networking and Internet Architecture', 
                    'cs.NE, Neural and Evolutionary Computing', 
                    'cs.NA, Numerical Analysis', 
                    'cs.OS, Operating Systems', 
                    'cs.OH, Other Computer Science',
                    'cs.PF, Performance', 
                    'cs.PL, Programming Languages', 
                    'cs.RO, Robotics', 
                    'cs.SI, Social and Information Networks', 
                    'cs.SE, Software Engineering',
                    'cs.SD, Sound',
                    'cs.SC, Symbolic Computation', 
                    'cs.SY, Systems and Control'
                ]
        else:
            label_list = [
                'cs.AI, Artificial Intelligence', 
                'cs.CL, Computation and Language', 
                'cs.CC, Computational Complexity', 
                'cs.CE, Computational Engineering, Finance, and Science', 
                'cs.CG, Computational Geometry', 
                'cs.GT, Computer Science and Game Theory', 
                'cs.CV, Computer Vision and Pattern Recognition', 
                'cs.CY, Computers and Society',
                'cs.CR, Cryptography and Security', 
                'cs.DS, Data Structures and Algorithms', 
                'cs.DB, Databases', 
                'cs.DL, Digital Libraries', 
                'cs.DM, Discrete Mathematics', 
                'cs.DC, Distributed, Parallel, and Cluster Computing', 
                'cs.ET, Emerging Technologies', 
                'cs.FL, Formal Languages and Automata Theory', 
                'cs.GL, General Literature', 
                'cs.GR, Graphics', 
                'cs.AR, Hardware Architecture', 
                'cs.HC, Human-Computer Interaction', 
                'cs.IR, Information Retrieval', 
                'cs.IT, Information Theory', 
                'cs.LO, Logic in Computer Science', 
                'cs.LG, Machine Learning', 
                'cs.MS, Mathematical Software', 
                'cs.MA, Multiagent Systems', 
                'cs.MM, Multimedia', 
                'cs.NI, Networking and Internet Architecture', 
                'cs.NE, Neural and Evolutionary Computing', 
                'cs.NA, Numerical Analysis', 
                'cs.OS, Operating Systems', 
                'cs.OH, Other Computer Science',
                'cs.PF, Performance', 
                'cs.PL, Programming Languages', 
                'cs.RO, Robotics', 
                'cs.SI, Social and Information Networks', 
                'cs.SE, Software Engineering',
                'cs.SD, Sound',
                'cs.SC, Symbolic Computation', 
                'cs.SY, Systems and Control'
            ]
    label2idx = {k: v for v, k in enumerate(label_list)}

    cnt = 0
    y, x = [], []
    for label, pred in zip(eval_decode_label, eval_pred):     
        if pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
            pred = pred[:-1]
        if pred.startswith(' '):
            pred = pred[1:]
        
        if args.graph_unsup:
            pred = pred.lower()
            label = label.lower()
            
        if pred not in label2idx.keys():
            cnt += 1
            continue
        y.append(label2idx[label])
        x.append(label2idx[pred])

    acc = accuracy_score(y, x)
    r = recall_score(y, x, average="macro")
    p = precision_score(y, x, average="macro")
    f1 = f1_score(y, x, average="macro")
    if args.graph_unsup:
        auc = roc_auc_score(y, x)
    else:
        auc = 0

    return acc, r, p, f1, auc, cnt / len(eval_decode_label)


def get_undirected_graph(edge_index):
    edge_index = torch.tensor(edge_index)
    row, col = edge_index[0], edge_index[1]
    g = dgl.graph((row, col))
    bg = dgl.to_bidirected(g)
    col, row = bg.edges()

    return torch.stack([row, col], dim=0)


def get_edge_num(x):
    return len(x['edge_index'][0])


def get_ans(x):
    ans = x["output"][:2] + '.' + x["output"][3:5].upper() + x["output"][5:]
    return ans


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    logprobs = params.get("logprobs", None)  # FIXME: Support logprobs>1.
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        if logprobs is not None:  # FIXME: Support logprobs for encoder-decoder models.
            raise NotImplementedError
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    else:
        start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values = out = None
    token_logprobs = [None]  # The first token has no logprobs.
    sent_interrupt = False
    finish_reason = None
    stopped = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(input_ids=start_ids, use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids],
                        device=device,
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)
        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "logprobs": ret_logprobs,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = "length"

    if stopped:
        finish_reason = "stop"

    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()

def stream_output(output_stream):
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    print(" ".join(output_text[pre:]), flush=True)
    return " ".join(output_text)