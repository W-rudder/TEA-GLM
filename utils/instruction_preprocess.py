import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from .utils import *
from .conversation import conv_templates, SeparatorStyle


IGNORE_TOKEN_ID = -100


def get_instructions(data_path):
    df = pd.read_json(data_path)
    df['edge_num'] = df.apply(get_edge_num, axis=1)
    df['gpt'] = df['output']
    df = df.reset_index(drop=True)
    df = df.sample(n=20, ignore_index=True)

    return df


def preprocess(instruction, tokenizer, max_length, mode='train'):
    conv = conv_templates["vicuna_v1_1"].copy()
    assert conv.sep_style == SeparatorStyle.TWO

    roles = conv.roles
    tokenizer.padding_side = 'right' if mode == 'train' else 'left'

    # Apply prompt templates
    conversations = []
    conv.append_message(roles[0], instruction["prompt"])
    # conv.append_message(roles[0], "What is the meaning of AI?")
    if mode == 'train':
        conv.append_message(roles[1], instruction["gpt"])
    else:
        conv.append_message(roles[1], None)
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    if mode != 'train':
        end_token = "</s>"
        targets = tokenizer(
            [instruction["gpt"]+end_token],
            return_tensors="pt",
            padding="max_length",
            max_length=100,
            truncation=True,
        ).input_ids
    else:
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct. the first label is not _, but _label
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            target[cur_len:] = IGNORE_TOKEN_ID


            if cur_len < max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        target_ids=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        text=conversations[0]
    )


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, args, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        if args.inference:
            self.instructions = get_instructions(f"./instruction/{args.test_dataset}/{args.test_dataset}_dataset_{self.mode}.json")
        else:
            self.instructions = get_instructions(f"./instruction/{args.dataset}/{args.dataset}_dataset_{self.mode}.json")

        self.paper = torch.load(f"./data/graph_data_paper.pt")
        self.ecom = torch.load(f"./data/graph_data_ecommercial.pt")
        self.name2data = {
            'arxiv': self.paper['arxiv'],
            'pubmed': self.paper['pubmed'],
            'cora': self.paper['cora'],
            'children': self.ecom['book_children'],
            'history': self.ecom['book_history'],
            'computer': self.ecom['computer'],
            'photo': self.ecom['photo'],
            'sports': self.ecom['sports'],
        }
        args.gnn_input = self.name2data[args.dataset].x.shape[1]
        print(f'Hidden dim: {args.gnn_input}')
        args.edge_dim = None
        
    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        raw = self.instructions.iloc[idx]
        instruction = raw.copy()
        tokens = " ".join([f"<Node {i}>" for i in range (1, 1 + self.args.num_token)])
        data_name = instruction['data']

        # remove abstract for citation dataset
        if data_name in ['children', 'history', 'computer', 'sports', 'photo']:
            instruction['prompt'] = raw['prompt'].replace("<Node 1>", tokens)
        else:
            instruction['prompt'] = (raw['prompt'].split('Abstract: ')[0] + 'Title: ' + raw['prompt'].split('Title: ')[1]).replace("<Node 1>", tokens)

        out_dict = preprocess(instruction, self.tokenizer, self.args.max_text_length, self.mode)
        
        # graph data
        graph = Data()
        graph.edge_index = torch.LongTensor(instruction['edge_index'])
        node_list = torch.LongTensor(instruction['node_set'])
        graph.x = self.name2data[data_name].x[node_list].to(dtype=torch.bfloat16)
        graph.lp = True if instruction['task'] == 'lp' else False
        graph.edge_attr = None
        is_node = (out_dict['input_ids'] >= 32000)

        out_dict['is_node'] = is_node
        out_dict['graph'] = graph

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        input_ids = []
        target_ids = []
        attention_mask = []
        is_node = []
        graph = []
        
        for i, entry in enumerate(batch):
            input_ids.append(entry['input_ids'])
            target_ids.append(entry['target_ids'])
            attention_mask.append(entry['attention_mask'])
            is_node.append(entry['is_node'])
            graph.append(entry['graph'])
        
        batch_entry['input_ids'] = torch.cat(input_ids, dim=0) # tensor
        batch_entry['target_ids'] = torch.cat(target_ids, dim=0) # tensor
        batch_entry['attn_mask']= torch.cat(attention_mask, dim=0) # tensor
        batch_entry['is_node'] = torch.cat(is_node, dim=0) # tensor
        batch_entry['graph'] = Batch.from_data_list(graph)

        return batch_entry      # Real batch data.