# Author: Jiaqi Yao

import torch
import logging
import torch.nn as nn

from transformers.utils import logging
from typing import Optional, List, Unpack
from transformers import AutoConfig, AutoModelForCausalLM
from model.utils import GenerationMixinForIH, LlamaIHConfig, LlamaIHModel
from transformers.models.llama.modeling_llama import KwargsForCausalLM, LlamaConfig, LlamaForCausalLM


logger = logging.get_logger(__name__)


class LlamaISEForCausalLM(LlamaForCausalLM, GenerationMixinForIH):
    config_class = LlamaIHConfig
    
    def __init__(self, config: LlamaConfig, ih_size=4):
        super(LlamaISEForCausalLM, self).__init__(config)
        assert ih_size > 0, "[***Error***]: ih_size must be positive integer!"
        config.ih_size = ih_size
        self.config = config
        self.ih_size=ih_size
        self.model = LlamaIHModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ise_emb = nn.Embedding(ih_size, config.hidden_size)
        
        logger.info("Instantiating LlamaISE model and initializing ise_emb")
        self.init_added_weight()
        self.post_init()
    
    def get_model(self):
        return self.model

    def init_added_weight(self):
        logger.warning("Initializing ise_emb")
        torch.nn.init.normal_(self.ise_emb.weight, mean=0.0, std=0.02)

    @classmethod
    def load_base_for_train(cls, pretrained_model_name_or_path, *model_args, **kwargs): # pass ih_ids in model_args
        logger.warning("Loading base model %s for training" % pretrained_model_name_or_path)
        # todo: add liger_kernel support, using `model_init_kwargs`
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs).to('cuda')
        model.train()
        model.init_added_weight()
        return model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ih_ids: Optional[torch.LongTensor] =None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):

        if ih_ids is not None:
            ise_embed = self.ise_emb(ih_ids)
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds + ise_embed
            
        return super().forward(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
    

AutoConfig.register("llama_ih", LlamaIHConfig)
AutoModelForCausalLM.register(LlamaIHConfig, LlamaISEForCausalLM)