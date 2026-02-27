# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import re
from copy import deepcopy
from torch.utils.data import Dataset
import logging
import io, json
from tqdm import tqdm
from config import PROMPT_FORMAT, IGNORE_ATTACK_SENTENCES, OTHER_DELM_FOR_TEST, OTHER_DELM_TOKENS, SPECIAL_DELM_TOKENS, DEFAULT_TOKENS, IGNORE_INDEX, TEXTUAL_DELM_TOKENS, DELIMITERS


def format_with_other_delimiters(text, test=False):
    test_idx = - OTHER_DELM_FOR_TEST
    mark = np.random.choice(OTHER_DELM_TOKENS['mark'][test_idx:] if test else OTHER_DELM_TOKENS['mark'][:test_idx]) + ':'
    
    def sample_delm(delm_name):
        role_name = 'user' if (delm_name == 'inst' or delm_name == 'inpt') else 'asst'
        if test: 
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][test_idx:]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][test_idx:])
        else:    
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][:test_idx]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][:test_idx])
        
        p = np.random.rand()
        if p < 1/3: return (role + delm).upper()
        elif p < 2/3: return (role + delm).lower()
        else: return role + delm
    
    for delm in DELIMITERS.values():
        text = text.replace(delm[0], mark.format(s=sample_delm('inst')))
        text = text.replace(delm[1], mark.format(s=sample_delm('inpt')))
        text = text.replace(delm[2], mark.format(s=sample_delm('resp')))
    return text

def generate_training_data(data_dicts, prompt_dict_name, attack, tokenizer):
    prompt_dict = PROMPT_FORMAT[prompt_dict_name]
    if attack == 'None':
        return [
            prompt_dict["prompt_input"].format_map(example) if example.get("input", "") != "" else prompt_dict["prompt_no_input"].format_map(example) for example in data_dicts
        ], [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]
    if attack == 'Completion':
        ref_inst_resp = {}
        for ref_sample in jload('data/alpaca_data.json'):  ref_inst_resp[ref_sample['instruction']] = ref_sample['output']
    sources = []

    for i in range(len(data_dicts)):
        # no anti-instruction tuning if there is no input
        if data_dicts[i].get("input", "") == "": sources.append(prompt_dict["prompt_no_input"].format_map(data_dicts[i]))
        else:
            injected_sample = deepcopy(np.random.choice(data_dicts)) 
            injected_sample['instruction'] = injected_sample['instruction']
            if injected_sample['instruction'][-1] == '?': 
                injected_prompt = 'answer the following question. ' + injected_sample['instruction'] + ' ' + injected_sample['input']
            else: 
                injected_prompt = injected_sample['instruction'][0].lower() + injected_sample['instruction'][1:] + ' ' + injected_sample['input']
            
            data_dicts_item = deepcopy(data_dicts[i])
            if data_dicts_item['input'][-1] != '.': data_dicts_item['input'] += '.'
            if attack == 'Naive':
                data_dicts_item['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Ignore':
                data_dicts_item['input'] += ' ' + np.random.choice(IGNORE_ATTACK_SENTENCES['train']) + ' ' + injected_prompt
            elif attack == 'Completion':
                data_dicts_item['input'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][2] + '\n' + ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + '\n\n' + \
                                                     DELIMITERS['SpclSpclSpcl'][0] + '\n' + injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['input'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][1] + '\n' + injected_sample['input']
                data_dicts_item['input'] = format_with_other_delimiters(data_dicts_item['input'], test=False)
            else: raise NotImplementedError

            sources.append(prompt_dict["prompt_input"].format_map(data_dicts_item))
    return sources, [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]


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

def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list] 
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _tokenize_fn_ih(strings, tokenizer, prompt_dict_name='SpclSpclSpcl'):
    """Tokenize a list of strings."""
    input_str = "[MARK] [INPT][COLN]\n" if prompt_dict_name == 'SpclSpclSpcl' else "### input:\n"
    resp_str = "[MARK] [RESP][COLN]\n" if prompt_dict_name == 'SpclSpclSpcl' else "### response:\n"

    tmp_1 = [s.split(resp_str) for s in strings]
    tmp_2 = [t[0].split(input_str) for t in tmp_1]
    inst = [t[0] for t in tmp_2]
    input = [''] * len(inst)
    for i, _ in enumerate(input):
        if len(tmp_1[i]) == 2:
            input[i] = (input[i] + input_str + tmp_2[i][1]) if len(tmp_2[i]) == 2 else input[i]
        elif len(tmp_1[i]) == 3:
            input[i] += input_str + tmp_2[i][1] + resp_str + tmp_1[i][1]
        else:
            print(tmp_1[i])
            raise NotImplementedError(f"Wrong examples data: not supported length: {len(tmp_1[i])}")
    
    inst_tokenized = tokenizer(
        inst,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
        padding=False,
        truncation=True
    )
    inst_lens = [len(inst_ele) for inst_ele in inst_tokenized.input_ids]
    input_tokenized = tokenizer(
        input,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
        padding=False,
        truncation=True
    )
    input_lens = [len(input_ele) for input_ele in input_tokenized.input_ids]
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm(strings, desc="Tokenizing text", total=len(strings))
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list] 
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    ih_ids = [torch.zeros_like(ids, dtype=torch.long) for ids in input_ids]
    for i in tqdm(range(len(input_ids)), desc="Seg ih_ids", total=len(input_ids)):
        ih_ids[i][inst_lens[i]+1: inst_lens[i]+input_lens[i]+1] = 1
        ih_ids[i][inst_lens[i]+input_lens[i]+1: ] = 2

    return dict(
        input_ids=input_ids,
        ih_ids=ih_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _partial_tokenize_fn(llm_input_dict_i, tokenizer, ih_size=3, delimiter='text'):
    """Tokenize a list of strings."""
    system = llm_input_dict_i['system'] if 'system' in llm_input_dict_i else ''
    strings = system + llm_input_dict_i['instruction'] + llm_input_dict_i['input']
    resp_str = '\n\n### response:\n' if delimiter == 'text' else '\n\n[MARK] [RESP][COLN]\n'
    tmp = strings.strip(resp_str) + '\n\n'
    tokenized = tokenizer(
        strings,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        add_special_tokens=True,
        # padding="max_length",
        truncation=True,
    )
    syst = tokenizer(system, return_tensors="pt", add_special_tokens=False).input_ids[0]
    syst_len = len(syst)
    inst = tokenizer(llm_input_dict_i['instruction'], return_tensors="pt", add_special_tokens=False).input_ids[0]
    inst_len = len(inst)
    inp = tokenizer(llm_input_dict_i['input'], return_tensors="pt", add_special_tokens=False).input_ids[0]
    inp_len = len(inp)
    tmp_ = tokenizer(tmp, return_tensors="pt", add_special_tokens=False).input_ids[0]
    tmp_len = len(tmp_)
    
    input_ids = labels = tokenized.input_ids[0]
    input_ids_lens = labels_lens = tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()

    # For the completion_real task, max_length should not be too small (the attack can be very long).
    # Otherwise input_ids will be truncated while inst/inp are not, which can break length assumptions.
    # assert len(input_ids) == inst_len + inp_len + 1
    ih_ids = torch.zeros_like(input_ids, dtype=torch.long)
    if ih_size == 3:
        ih_ids[syst_len + inst_len + 1: ] = 1
        ih_ids[tmp_len + 1: ] = 2
    elif ih_size == 4:
        ih_ids[syst_len + 1: syst_len + inst_len + 1] = 1
        ih_ids[syst_len + inst_len + 1: ] = 2
        ih_ids[tmp_len + 1: ] = 3
    else:
        raise NotImplementedError("[***Error***]: ih_size must in (3, 4)!")

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i, (token, token_id) in enumerate(zip(tokens, input_ids)):
        ih_id = ih_ids[i]
        logging.info(f"( {token}, {token_id}, {ih_id} )")

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        ih_ids=ih_ids,
    )


def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def preprocess_ih(sources, targets, tokenizer, prompt_dict_name='SpclSpclSpcl'):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn_ih(strings, tokenizer, prompt_dict_name) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    ih_ids = examples_tokenized["ih_ids"]
    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, ih_ids=ih_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, attack, downsample=True):
        super(SupervisedDataset, self).__init__() 
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        prompt_dict_name, attacks = attack.split('_')
        source_clean, targets_clean = generate_training_data(list_data_dict, prompt_dict_name, 'None', tokenizer)
        
        if attacks == 'None': 
            sources, targets = source_clean, targets_clean
            self.data_copy_count = 1
        else:
            attacks = re.findall('[A-Z][^A-Z]*', attacks)
            sources = []; targets = []
            self.data_copy_count = len(attacks) + len(attacks) * downsample
            
            for a in attacks:
                source, target = generate_training_data(list_data_dict, prompt_dict_name, a, tokenizer)
                sources += source; targets += target
                if downsample: sources += source_clean; targets += targets_clean
                    
            # downsize data to original size with 50% clean data
            if downsample:
                sample_batch_id = np.random.choice(range(self.data_copy_count), len(source_clean))
                sample_id = [(x * len(sample_batch_id) + i) for i, x in enumerate(sample_batch_id)]
                sources = np.array(sources)[sample_id].tolist(); targets = np.array(targets)[sample_id].tolist()
            else:
                sources = np.array(sources).tolist(); targets = np.array(targets).tolist()

        logging.warning("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"] 

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i): return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SupervisedDatasetIH(Dataset):
    def __init__(self, data_path: str, tokenizer, attack, downsample=True):
        super(SupervisedDatasetIH, self).__init__() 
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        prompt_dict_name, attacks = attack.split('_')
        source_clean, targets_clean = generate_training_data(list_data_dict, prompt_dict_name, 'None', tokenizer)
        
        if attacks == 'None': 
            sources, targets = source_clean, targets_clean
            self.data_copy_count = 1
            # with open('/home/jiaqiyao/dev/code/paper_code/data/struq/sources_delm.json', 'w', encoding='utf-8') as f:
            #     json.dump(sources, f, ensure_ascii=False, indent=4)
            # with open('/home/jiaqiyao/dev/code/paper_code/data/struq/targets_delm.json', 'w', encoding='utf-8') as f:
            #     json.dump(targets, f, ensure_ascii=False, indent=4)
        else:
            attacks = re.findall('[A-Z][^A-Z]*', attacks)
            sources = []; targets = []
            self.data_copy_count = len(attacks) + len(attacks) * downsample
            
            for a in attacks:
                source, target = generate_training_data(list_data_dict, prompt_dict_name, a, tokenizer)
                sources += source; targets += target
                if downsample: sources += source_clean; targets += targets_clean
                    
            # downsize data to original size with 50% clean data
            if downsample:
                sample_batch_id = np.random.choice(range(self.data_copy_count), len(source_clean))
                sample_id = [(x * len(sample_batch_id) + i) for i, x in enumerate(sample_batch_id)]
                sources = np.array(sources)[sample_id].tolist(); targets = np.array(targets)[sample_id].tolist()
                '''
                with open('/home/jiaqiyao/dev/code/paper_code/data/struq/sources_delm_adv.json', 'w', encoding='utf-8') as f:
                    json.dump(sources, f, ensure_ascii=False, indent=4)
                with open('/home/jiaqiyao/dev/code/paper_code/data/struq/targets_delm_adv.json', 'w', encoding='utf-8') as f:
                    json.dump(targets, f, ensure_ascii=False, indent=4)
                '''
            else:
                sources = np.array(sources).tolist(); targets = np.array(targets).tolist()

        logging.warning("Tokenizing inputs...")
        data_dict = preprocess_ih(sources, targets, tokenizer, prompt_dict_name)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"] 
        self.ih_ids = data_dict["ih_ids"]

        # for i in range(len(self.input_ids)):
        #     tokens = tokenizer.convert_ids_to_tokens(self.input_ids[i])
        #     for j, (token, token_id) in enumerate(zip(tokens, self.input_ids[i])):
        #         ih_id = self.ih_ids[i][j]
        #         logging.info(f"Sample {i}: ( {token}, {token_id}, {ih_id} )")

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i): return dict(input_ids=self.input_ids[i], ih_ids=self.ih_ids[i], labels=self.labels[i])