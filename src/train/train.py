import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import trl
import torch
import random
# import logging
import numpy as np
import transformers

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from struq import SupervisedDataset, SupervisedDatasetIH
from config import IGNORE_INDEX, DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS
from transformers import Trainer
from transformers.utils import logging, is_sagemaker_mp_enabled
import torch.nn as nn
import bitsandbytes

from model import LlamaICAQformerForCausalLM
from model_nocrossmask import LlamaICAQformerForCausalLM_NoCrossMask
from model_selfmask import LlamaICAQformerForCausalLM_SelfMask

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    window_size: int = field(default=0, metadata={"help": "Window size for the sliding window attention."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for tokenization."})
    paper_model: str = field(default="struq")
    delitimer: str = field(default="text")
    train_type: Optional[str] = field(default="sft")
    ih_size: Optional[int] = field(default=-1)
    ICA_num_attention_heads: Optional[int] = field(default=8)
    adapter_dim: Optional[int] = field(default=256)
    attn_implementation: Optional[str] = field(default='fdpa')
    add_special_tokens: Optional[bool] = field(default=True)
    freeze_model: Optional[bool] = field(default=False)




@dataclass
class DataArguments: data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class AttackArguments: 
    attack: str = field(default='TextTextText_None', metadata={"help": "Attack type for SFT/Align"})
    alignment: str = field(default='none', metadata={"help": "Alignment type."})


@dataclass
class TrainingArguments(trl.ORPOConfig):#transformers.TrainingArguments): # 
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    learning_rate_factor: Optional[float] = field(default=1.0)
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)
    beta: float = field(default=0.1)
    ref_model_init_kwargs: Optional[str] = field(default=None)
    precompute_ref_log_probs: Optional[bool] = field(default=False)
    desirable_weight: Optional[float] = field(default=1)
    undesirable_weight: Optional[float] = field(default=1)
    # max_seq_length: Optional[int] = field(default=512)
    data_seed: Optional[int] = field(default=42)
    wandb_project: Optional[str] = field(
        default="default_project",
        metadata={"help": "WandB project name."}
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForSupervisedDatasetIH(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, ih_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "ih_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        ih_ids = torch.nn.utils.rnn.pad_sequence(ih_ids, batch_first=True, padding_value=2)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            ih_ids=ih_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class LayerTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer, allowing different LR for different parts of the model.
        We also keep the special logic for bitsandbytes (Adam8bit) and sagemaker mp.
        """
        # 0) Only create the optimizer if it hasn't been created yet.
        if self.optimizer is not None:
            return self.optimizer

        # 1) Get the underlying model (SageMaker MP compatible).
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        # 2) Get parameter names that should use weight decay (excluding LN, bias, etc.).
        decay_parameters = self.get_decay_parameter_names(opt_model)

        # 3) Set different learning rates for different parameter groups as needed.
        #   Example: group 1 if name.startswith("model.") (or "lm_head."), otherwise group 2.
        #   Also keep the decay vs no_decay split for weight decay.
        param_optimizer = list(opt_model.named_parameters())

        # 3.1 First split into "model." vs "other" (for different LR),
        #     then within each group split into "decay" vs "no_decay" (for weight_decay).
        group1_decay = {
            "params": [
                p for n, p in param_optimizer
                if p.requires_grad and (n in decay_parameters) and (n.startswith("model.") or n.startswith("lm_head."))
            ],
            "weight_decay": self.args.weight_decay,
            "lr": self.args.learning_rate,  # e.g., base learning rate
        }
        group1_no_decay = {
            "params": [
                p for n, p in param_optimizer
                if p.requires_grad and (n not in decay_parameters) and (n.startswith("model.") or n.startswith("lm_head."))
            ],
            "weight_decay": 0.0,
            "lr": self.args.learning_rate,
        }

        group2_decay = {
            "params": [
                p for n, p in param_optimizer
                if p.requires_grad and (n in decay_parameters) and not n.startswith("model.") and not n.startswith("lm_head.")
            ],
            "weight_decay": self.args.weight_decay,
            "lr": self.args.learning_rate * self.args.learning_rate_factor,
        }
        group2_no_decay = {
            "params": [
                p for n, p in param_optimizer
                if p.requires_grad and (n not in decay_parameters) and not n.startswith("model.") and not n.startswith("lm_head.")
            ],
            "weight_decay": 0.0,
            "lr": self.args.learning_rate * self.args.learning_rate_factor,
        }

        optimizer_grouped_parameters = [
            group1_decay,
            group1_no_decay,
            group2_decay,
            group2_no_decay,
        ]

        # 4) Check whether optimizer_cls_and_kwargs was already specified.
        #    If not, fall back to get_optimizer_cls_and_kwargs.
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        
        logger.warning(f"optimizer_cls: {optimizer_cls}, optimizer_kwargs: {optimizer_kwargs}")

        # 5) If optimizer_kwargs contains "params"/"model"/"optimizer_dict", it overrides the default grouping.
        if "params" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("params")
        if "model" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("model")
        if "optimizer_dict" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

        # 6) Build the final optimizer.
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # 7) For Adam8bit, keep bitsandbytes' special handling for embedding parameters.
        if optimizer_cls.__name__ == "Adam8bit":
            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped/2**20}M params")

        # 8) Under SageMaker MP, wrap as a distributed optimizer.
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    

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
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    REAL_DELIMITERS_INIT_EMBD_IND, _ = get_embedding_indices(tokenizer)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens] = input_embeddings_avg
        output_embeddings[-num_new_tokens] = output_embeddings_avg

        for i in range(len(SPECIAL_DELM_TOKENS)): ### initialize real delimiter's embedding by the existing ones
            input_embeddings[-num_new_tokens+i+1] = input_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]
            output_embeddings[-num_new_tokens+i+1] = output_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, downsample=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, attack=data_args.attack, downsample=downsample)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_supervised_data_module_for_ih(tokenizer: transformers.PreTrainedTokenizer, data_args, downsample=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDatasetIH(tokenizer=tokenizer, data_path=data_args.data_path, attack=data_args.attack, downsample=downsample)
    data_collator = DataCollatorForSupervisedDatasetIH(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))

    # =================================================================
    # Switch argument parsing mode (supports config file input).
    # model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    if os.sys.argv[-1].endswith(('.yml', '.yaml', '.json')):
        # If a config file is provided, parse arguments from the file
        model_args, data_args, training_args, attack_args = parser.parse_yaml_file(os.sys.argv[-1])
    else:
        # Otherwise, parse arguments from the command line
        model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()

    data_args.attack = attack_args.attack 
    if 'Instruct' in model_args.model_name_or_path: assert 'SpclSpclSpcl' not in data_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir + '/logs/')
    # setup_logging(training_args.output_dir)
    set_seed(training_args.seed)

    rank = int(os.getenv("RANK", "0"))
    if "wandb" in training_args.report_to and rank == 0:
        import wandb
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.run_name if training_args.run_name else None,
            config={
                "model_name": model_args.model_name_or_path,
                "batch_size": training_args.per_device_train_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
                # Add other hyperparameters you want to track
            }
        )

    # =================================================================
    if model_args.paper_model == 'struqbaseline':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'ise':
        model = LlamaISEForCausalLM.load_base_for_train(
            model_args.model_name_or_path,
            model_args.ih_size,
            # use_liger=False,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'isesinusoidal':
        model = LlamaIsinSEForCausalLM.load_base_for_train(
            model_args.model_name_or_path,
            model_args.ih_size,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'isesinusoidalandlearn':
        model = LlamaIsinSlearnEForCausalLM.load_base_for_train(
            model_args.model_name_or_path,
            model_args.ih_size,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'learnablesinise':
        model = LlamaLearnableSinISEForCausalLM.load_base_for_train(
            model_args.model_name_or_path,
            model_args.ih_size,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'icaformer':
        model = LlamaICAformerForCausalLM.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'icaformer_adapter':
        model = LlamaICAAdapterformerForCausalLM.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            model_args.adapter_dim,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'icaseqQformer':
        model = LlamaICAQformerForCausalLM.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'icaseqQformernocrossmask':
        model = LlamaICAQformerForCausalLM_NoCrossMask.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'icaseqQformerselfmask':
        model = LlamaICAQformerForCausalLM_SelfMask.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'icaseqQformernoise':
        model = LlamaICAQformerNoiseForCausalLM.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'icaseqQformernoself':
        model = LlamaICAQformerNoSelfForCausalLM.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
    elif model_args.paper_model == 'ica':
        model = LlamaICAForCausalLM.init_from_pretrained(
            model_args.model_name_or_path,
            model_args.ih_size,
            model_args.ICA_num_attention_heads,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    if model_args.freeze_model:
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")
            if name.startswith("model."):
                param.requires_grad = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token'] ###
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS ### 
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    # =================================================================
    # Switch IH dataset construction.
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample)
    if 'baseline' in model_args.paper_model:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample)
    else:
        data_module = make_supervised_data_module_for_ih(tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    if training_args.learning_rate_factor != 1.0:
        trainer = LayerTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
            )
    else:
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    if "wandb" in training_args.report_to and rank == 0:
        wandb.finish()

    
if __name__ == "__main__":
    train()