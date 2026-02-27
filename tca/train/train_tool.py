# Author: Jiaqi Yao

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trl
import torch
import random
import logging
import numpy as np

from typing import Dict, Optional
from dataclasses import dataclass, field
from model.ise_model import LlamaISEForCausalLM
from model.cahl_model import LlamaCAHLForCausalLM
from train.struq_tool import SupervisedDatasetForTool
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from train.config import IGNORE_INDEX, DEFAULT_TOKENS, TEXTUAL_DELM_TOKENS
from transformers import (
    HfArgumentParser,
    Trainer,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


@dataclass
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.1-8B")
    window_size: int = field(default=0, metadata={"help": "Window size for the sliding window attention."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for tokenization."})
    model_type: str = field(default="struq")
    train_type: Optional[str] = field(default="sft")
    ih_size: Optional[int] = field(default=-1)
    attn_implementation: Optional[str] = field(default='fdpa')
    add_special_tokens: Optional[bool] = field(default=True)
    wandb_project: Optional[str] = field(
        default="default_project",
        metadata={"help": "WandB project name."}
    )
    cahl_attention_heads: Optional[int] = field(default=1)
    num_labels: Optional[int] = field(default=3)
    Qformer_hidden_size: Optional[int] = field(default=256)
    Qformer_num_heads: Optional[int] = field(default=4)
    chat_template: str = field(default="tool_chat_template_llama3.1_json.jinja")
    

@dataclass
class DataArguments: data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(trl.ORPOConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)
    beta: float = field(default=0.1)
    ref_model_init_kwargs: Optional[str] = field(default=None)
    precompute_ref_log_probs: Optional[bool] = field(default=False)
    desirable_weight: Optional[float] = field(default=1)
    undesirable_weight: Optional[float] = field(default=1)
    # max_seq_length: Optional[int] = field(default=512)
    data_seed: Optional[int] = field(default=42)
    use_liger: Optional[bool] = field(default=False)


@dataclass
class DataCollatorForSupervisedDatasetIH(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer
    ih: bool

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        if self.ih:
            input_ids, ih_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "ih_ids", "labels"))
        else:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        ih_ids = torch.nn.utils.rnn.pad_sequence(ih_ids, batch_first=True, padding_value=2) if self.ih else None
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if self.ih:
            return dict(
                input_ids=input_ids,
                ih_ids=ih_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'logs/train.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_embedding_indices(tokenizer):
    init_values = [tokenizer.encode(v, add_special_tokens=False)[0] for v in TEXTUAL_DELM_TOKENS]
    ignore_values = [i for i in range(len(tokenizer)) if tokenizer.decode(i) == "#"]
    return init_values, ignore_values


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens] = input_embeddings_avg
        output_embeddings[-num_new_tokens] = output_embeddings_avg

        for i in range(num_new_tokens):
            input_embeddings[-num_new_tokens+i+1] = input_embeddings_avg
            output_embeddings[-num_new_tokens+i+1] = output_embeddings_avg


def make_supervised_data_module_for_tool(tokenizer: PreTrainedTokenizer, data_args, ih=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDatasetForTool(tokenizer=tokenizer, data_path=data_args.data_path, ih=ih)
    data_collator = DataCollatorForSupervisedDatasetIH(tokenizer=tokenizer, ih=ih)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if '--cfg' in os.sys.argv:
        # If a config file is provided, parse arguments from the file
        model_args, data_args, training_args = parser.parse_yaml_file(os.sys.argv[-1])
    else:
        # Otherwise, parse arguments from the command line
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('\n\n' + training_args.output_dir + '\n\n')

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir + '/logs/')
    setup_logging(training_args.output_dir)
    set_seed(training_args.data_seed)

    rank = int(os.getenv("RANK", "0"))
    if "wandb" in training_args.report_to and rank == 0:
        import wandb
        wandb.init(
            project=model_args.wandb_project,
            name=training_args.run_name if training_args.run_name else None,
            config={
                "model_name": model_args.model_name_or_path,
                "batch_size": training_args.per_device_train_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
            }
        )

    # device_map = {'model.embed_tokens': 0}
    # for i in range(16):
    #     device_map[f"model.layers.{i}"] = 0
    # for i in range(16, 32):
    #     device_map[f"model.layers.{i}"] = 1
    # device_map["lm_head"] = 1
    # device_map["model.norm"] = 1
    
    if model_args.model_type == 'struq':
        if training_args.use_liger:
            print("======== using liger_kernel")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                attn_implementation=model_args.attn_implementation,
                # device_map=device_map,
            ).to('cuda').train()
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                attn_implementation=model_args.attn_implementation,
                # device_map=device_map,
            ).to('cuda').train()
    elif model_args.model_type == 'ise':
        # device_map["ise_emb"] = 0
        model = LlamaISEForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
            # device_map=device_map,
        ).to('cuda').train()
    elif model_args.model_type == 'cahl':
        print("====== cahl")
        model = LlamaCAHLForCausalLM.init_from_pretrained(
            model_args.model_name_or_path,
            ih_size=model_args.ih_size,
            cahl_attention_heads=model_args.cahl_attention_heads,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        ).to('cuda').train()
    else:
        raise NotImplementedError("[***Error***]: Other models are not supported!")


    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    with open(model_args.chat_template, 'r', encoding='utf-8') as f:
        chat_template = f.read()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        chat_template=chat_template
    )

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token']
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    data_module = make_supervised_data_module_for_tool(tokenizer=tokenizer, data_args=data_args, ih=model_args.paper_model != 'struq')

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    
if __name__ == "__main__":
    train()