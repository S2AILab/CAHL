# Author: StruQ Team
# Modified by Jiaqi Yao

import re
import torch
import numpy as np
import logging
import io, json

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset
from train.config import IGNORE_INDEX


def generate_training_data_for_tool(data_dicts, tokenizer):
    logging.info(f"{tokenizer.apply_chat_template(data_dicts[0]['messages'], tokenize=False)}{tokenizer.eos_token}")
    return [f"{tokenizer.apply_chat_template(data['messages'], tokenize=False)}{tokenizer.eos_token}" for data in data_dicts]


def jload(f, mode="r"):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    if isinstance(obj, (dict, list)): json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str): f.write(obj)
    else: raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def _tokenize_fn_tool(examples, tokenizer, ih=True):
    sys_pattern = re.compile(r'<\|begin_of_text\|><\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)<\|eot_id\|>', re.DOTALL)
    msg_pattern = re.compile(r'<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)<\|eot_id\|>', re.DOTALL)

    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        )
        for text in tqdm(examples, desc="Tokenizing text", total=len(examples))
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    labels = deepcopy(input_ids)
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    if ih:
        ih_ids = [torch.zeros_like(ids, dtype=torch.long) for ids in input_ids]
        for ex_idx, text in enumerate(examples):
            sys_matches = sys_pattern.findall(text)
            msg_matches = msg_pattern.findall(text)

            result = []
            for role, content in sys_matches:
                segment = f'<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>'
                result.append(segment)
            for role, content in msg_matches[1: ]:
                segment = f'<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>'
                result.append(segment)
            
            idx = 0
            for segment in result:
                token_ids = tokenizer(
                    segment,
                    max_length=tokenizer.model_max_length,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False
                ).input_ids
                last = idx
                idx += len(token_ids)
                if '<|start_header_id|>system<|end_header_id|>' in segment or '<|start_header_id|>user<|end_header_id|>' in segment:
                    ih_ids[ex_idx][last: idx] = 0
                    labels[ex_idx][last: idx] = IGNORE_INDEX
                elif '<|start_header_id|>tool<|end_header_id|>' in segment:
                    ih_ids[ex_idx][last: idx] = 1
                    labels[ex_idx][last: idx] = IGNORE_INDEX
                elif '<|start_header_id|>assistant<|end_header_id|>' in segment:
                    ih_ids[ex_idx][last: idx] = 2
            ih_ids[ex_idx][idx: ] = 2
            labels[ex_idx][: last - 1] = IGNORE_INDEX
            labels[ex_idx][idx + 1: ] = IGNORE_INDEX

            # ========== print in log for debugging ==========
            # for idss in input_ids[ex_idx]:
            #     logging.info(f"input_id: {idss}")
            # tokens = tokenizer.convert_ids_to_tokens(input_ids[ex_idx])
            # for i, (token, token_id) in enumerate(zip(tokens, input_ids[ex_idx])):
            #     ih_id = ih_ids[ex_idx][i]
            #     logging.info(f"( {token}, {token_id}, {ih_id} )")

    return dict(
        input_ids=input_ids,
        ih_ids=ih_ids if ih else None,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_tool(examples, tokenizer, ih=True):
    examples_tokenized = _tokenize_fn_tool(examples, tokenizer, ih)
    input_ids = examples_tokenized["input_ids"]
    ih_ids = examples_tokenized["ih_ids"] if ih else None
    labels = examples_tokenized["labels"]
    # for label, source_len in zip(labels, examples_tokenized["input_ids_lens"]):
    #     label[: source_len] = IGNORE_INDEX
    return dict(
        input_ids=input_ids,
        ih_ids=ih_ids,
        labels=labels
    )


class SupervisedDatasetForTool(Dataset):
    def __init__(self, data_path: str, tokenizer, ih=True):
        super(SupervisedDatasetForTool, self).__init__()
        self.ih = ih
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        examples = generate_training_data_for_tool(list_data_dict, tokenizer)

        logging.warning("Tokenizing inputs...")
        data_dict = preprocess_tool(examples, tokenizer, ih)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"] 
        self.ih_ids = data_dict["ih_ids"] if ih else None

        if ih:
            for i in range(5):
                tokens = tokenizer.convert_ids_to_tokens(self.input_ids[i])
                for j, (token, token_id, label) in enumerate(zip(tokens, self.input_ids[i], self.labels[i])):
                    ih_id = self.ih_ids[i][j]
                    logging.info(f"Sample {i}: ( {token}, {token_id}, {label}, {ih_id} )")
                logging.info("-----------------------------------------------------")
        
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i): return dict(input_ids=self.input_ids[i], labels=self.labels[i], ih_ids=self.ih_ids[i] if self.ih else None)