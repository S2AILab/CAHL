import torch
import logging
import torch.nn as nn

from typing import Union
from dataclasses import dataclass
from utils import GenerationMixinForIH, LlamaIHConfig, LlamaIHModel
from utils_cache import GenerationMixinForQformer
from typing import Optional, List, Unpack, Tuple
from transformers.utils import ModelOutput, logging
from transformers.models.llama.modeling_llama import KwargsForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaRotaryEmbedding, apply_rotary_pos_emb
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.cache_utils import Cache, DynamicCache
logger = logging.get_logger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F


class QformerCrossAttention(nn.Module):
    """
    Custom cross-attention used with (query=ih_query, key=inputs_embeds, value=inputs_embeds),
    with support for a custom Qformer KV cache.
    """
    def __init__(self, embed_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {embed_dim} and {num_heads}).")

        # Projection matrices: Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,             # (batch_size, seq_len_q, hidden_size)
        key_value: torch.Tensor,         # (batch_size, seq_len_k, hidden_size)
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        Qformer_past_key_values: Optional[Union[Cache, dict]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        gradient_checkpointing: Optional[bool] = False, 
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        """
        Args:
            query: Tensor shaped like [batch_size, seq_len_q, embed_dim].
            key_value: Tensor shaped like [batch_size, seq_len_k, embed_dim], used as K,V for cross-attention.
            attn_mask: Boolean mask shaped like [B * num_heads, seq_len_q, seq_len_k].
                       True = masked, False = visible.
            use_cache: Whether to enable caching.
            Qformer_past_key_values: When caching is enabled, cross-attention K,V are stored/retrieved here.
            cache_key: Key name used inside Qformer_past_key_values to distinguish cross-attention entries.
        Returns:
            attn_output: [batch_size, seq_len_q, embed_dim]
            updated_Qformer_past_key_values: Updated cache (when use_cache=True).
        """
        if gradient_checkpointing and self.training and use_cache:
            logging.info(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        return_legacy_cache = False
        if use_cache and not isinstance(Qformer_past_key_values, Cache):
            return_legacy_cache = True
            if Qformer_past_key_values is None:
                Qformer_past_key_values = DynamicCache()
            else:
                Qformer_past_key_values = DynamicCache.from_legacy_cache(Qformer_past_key_values)
                logging.info(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        # 1) Project Q, K, V.
        bsz_q, len_q, _ = query.size()
        bsz_k, len_k, _ = key_value.size()

        # (batch_size, seq_len_q, embed_dim)
        q = self.q_proj(query)  
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # 2) If caching is enabled, append past K,V; 3) store the new K,V back into Qformer_past_key_values.
        #    We assume Qformer_past_key_values is a Cache/dict-like object.
        if Qformer_past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            k, v = Qformer_past_key_values.update(k, v, self.layer_idx, cache_kwargs)


        # 4) Reshape into multi-head form: [bsz, seq_len, num_heads, head_dim] -> [bsz * num_heads, seq_len, head_dim]
        def shape_proj(x: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, _ = x.size()
            x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
            x = x.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, seq_len, self.head_dim)
            return x

        q_ = shape_proj(q)
        k_ = shape_proj(k)
        v_ = shape_proj(v)

        # Note: attn_mask is shaped [bsz * num_heads, seq_len_q, seq_len_k] and must align with q_/k_ batch dimension.
        # If attn_mask is None, that's also fine.

        # 5) Use PyTorch 2.x native scaled_dot_product_attention.
        attn_output_ = F.scaled_dot_product_attention(
            query=q_, key=k_, value=v_,
            attn_mask=attn_mask,  # bool mask: True = masked; SDPA sets these positions to -inf
            dropout_p=0.0,        # add dropout here if needed
            # is_causal=True      # we use a custom mask instead of the default causal mask
        )

        # 6) Merge heads back: [bsz, seq_len_q, embed_dim]
        def merge_heads(x: torch.Tensor) -> torch.Tensor:
            bnh, sqlen, hdim = x.size()
            b = bnh // self.num_heads
            x = x.reshape(b, self.num_heads, sqlen, hdim).permute(0, 2, 1, 3).reshape(b, sqlen, self.num_heads * hdim)
            return x

        attn_output = merge_heads(attn_output_)
        # 7) Final output projection.
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if use_cache:
            if return_legacy_cache:
                Qformer_past_key_values = Qformer_past_key_values.to_legacy_cache()
        else:
            Qformer_past_key_values = None
        outputs += (Qformer_past_key_values,)
        return outputs
    
class QformerSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {embed_dim} and {num_heads}).")

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,       # [batch_size, seq_len, hidden_size], used as Q as well as K,V
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        Qformer_past_key_values: Optional[Union[Cache, dict]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        gradient_checkpointing: Optional[bool] = False,
    ):
        if gradient_checkpointing and self.training and use_cache:
            logging.info(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        return_legacy_cache = False
        if use_cache and not isinstance(Qformer_past_key_values, Cache):
            return_legacy_cache = True
            if Qformer_past_key_values is None:
                Qformer_past_key_values = DynamicCache()
            else:
                Qformer_past_key_values = DynamicCache.from_legacy_cache(Qformer_past_key_values)
                logging.info(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        # 1) Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if Qformer_past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            k, v = Qformer_past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        

        # reshape q,k,v to multi-head
        def shape_proj(x: torch.Tensor):
            bsz, seq_len, _ = x.size()
            x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
            x = x.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, seq_len, self.head_dim)
            return x

        q_ = shape_proj(q)
        k_ = shape_proj(k)
        v_ = shape_proj(v)

        attn_output_ = F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=attn_mask, dropout_p=0.0
        )

        # merge
        def merge_heads(x: torch.Tensor):
            bnh, sqlen, hdim = x.size()
            b = bnh // self.num_heads
            x = x.reshape(b, self.num_heads, sqlen, hdim).permute(0, 2, 1, 3).reshape(b, sqlen, self.num_heads * hdim)
            return x

        attn_output = merge_heads(attn_output_)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if use_cache:
            if return_legacy_cache:
                Qformer_past_key_values = Qformer_past_key_values.to_legacy_cache()
        else:
            Qformer_past_key_values = None
        outputs += (Qformer_past_key_values,)

        return attn_output, Qformer_past_key_values if use_cache else None



@dataclass
class QformerOutputWithPast(ModelOutput):
    """
        Extended output structure for forward() that returns:
            - Llama past_key_values
            - Our custom Qformer past_key_values
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    Qformer_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class LlamaICAQformerForCausalLM(LlamaForCausalLM, GenerationMixinForQformer):
    config_class = LlamaIHConfig

    def __init__(self, config: LlamaConfig, ih_size, ICA_num_attention_heads):
        super().__init__(config)
        self.model = LlamaIHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        config.ih_size = ih_size
        config.ICA_num_attention_heads = ICA_num_attention_heads
        self.ih_size = ih_size    
        self.ICA_num_attention_heads = ICA_num_attention_heads
        logger.warning('ICA_num_attention_heads: %d' % ICA_num_attention_heads)
        # segment embeddings
        self.ise_emb = nn.Embedding(config.ih_size, config.hidden_size)
        self.ih_query = nn.Embedding(config.ih_size, config.hidden_size)

        self.alpha_role = nn.Parameter(torch.zeros(config.ih_size))

        self.cross_attention = QformerCrossAttention(
            embed_dim=config.hidden_size,
            num_heads=ICA_num_attention_heads,
            layer_idx=0
        )
        self.self_attention = QformerSelfAttention(
            embed_dim=config.hidden_size,
            num_heads=ICA_num_attention_heads,
            layer_idx=1
        )

    @classmethod
    def init_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Initialize cross/self attention projections explicitly.
        logger.warning("Loading base model %s for training" % pretrained_model_name_or_path)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        logger.warning("Initializing ise_emb")
        logger.warning("Initializing attention weights")
        with torch.no_grad():
            nn.init.normal_(model.ise_emb.weight, mean=0.0, std=model.config.initializer_range)
            nn.init.normal_(model.ih_query.weight, mean=0.0, std=model.config.initializer_range)

            nn.init.normal_(model.cross_attention.q_proj.weight, mean=0.0, std=model.config.initializer_range)
            nn.init.normal_(model.cross_attention.k_proj.weight, mean=0.0, std=model.config.initializer_range)
            nn.init.normal_(model.cross_attention.v_proj.weight, mean=0.0, std=model.config.initializer_range)
            nn.init.normal_(model.cross_attention.out_proj.weight, mean=0.0, std=model.config.initializer_range)

            nn.init.normal_(model.self_attention.q_proj.weight, mean=0.0, std=model.config.initializer_range)
            nn.init.normal_(model.self_attention.k_proj.weight, mean=0.0, std=model.config.initializer_range)
            nn.init.normal_(model.self_attention.v_proj.weight, mean=0.0, std=model.config.initializer_range)
            nn.init.normal_(model.self_attention.out_proj.weight, mean=0.0, std=model.config.initializer_range)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ih_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        Qformer_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        gradient_checkpointing: Optional[bool] = False,
        **kwargs,
    ):
        """
        Adds Qformer_past_key_values to store incremental KV for cross/self attention.
        """
        # 1) Handle gradient_checkpointing vs use_cache compatibility.
        if gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # 2) Legacy-format compatibility for Qformer_past_key_values.
        return_legacy_cache = False
        if use_cache and not isinstance(Qformer_past_key_values, Cache) and Qformer_past_key_values is not None:
            # User provided a legacy tuple.
            return_legacy_cache = True
            Qformer_past_key_values = DynamicCache.from_legacy_cache(Qformer_past_key_values)
            logger.warning(
                "Detected legacy Qformer_past_key_values format. Will convert to `DynamicCache` structure."
            )
        # 3) Compute ih_embed / ih_query etc.
        if ih_ids is not None:
            ih_embed = self.ise_emb(ih_ids)
            ih_query = self.ih_query(ih_ids)
            alpha = self.alpha_role[ih_ids]

            # Prefer provided inputs_embeds; otherwise use embed_tokens.
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)

            # -- Keep your segment-wise mask construction logic here --
            # Note: SDPA uses a boolean mask where True means "masked" (i.e., not attended).
            # Be careful: this differs from some other attention implementations.
            # (For simplicity we keep this logic unchanged. For incremental decoding, building a B×L×L mask
            #  at every step can be expensive and may require a more sophisticated partial update strategy.)
            batch_size, seq_len = ih_ids.shape
            causal_mask = torch.zeros(
                (batch_size, seq_len, seq_len), device=inputs_embeds.device, dtype=torch.bool
            )
            change_indices_list = [
                (ih_ids[b] != ih_ids[b].roll(1, 0)).nonzero(as_tuple=True)[0].tolist() for b in range(batch_size)
            ]
            change_indices_list = [[0] + indices for indices in change_indices_list]
            for b in range(batch_size):
                prev_idx = change_indices_list[b][0]
                for idx in change_indices_list[b][1:]:
                    causal_mask[b, prev_idx:idx, prev_idx:idx] = True
                    prev_idx = idx
                causal_mask[b, prev_idx:seq_len, prev_idx:seq_len] = True

            causal_mask = causal_mask.unsqueeze(1)  # [B, 1, L, L]
            mask_repeated = causal_mask.repeat(1, self.ICA_num_attention_heads, 1, 1)
            causal_mask = mask_repeated.view(batch_size*self.ICA_num_attention_heads, seq_len, seq_len)


            # 4) Cross-attention: Q=ih_query, K=V=inputs_embeds
            cross_attention_output, Qformer_past_key_values = self.cross_attention(
                query=ih_query,
                key_value=inputs_embeds,
                attn_mask=causal_mask,
                use_cache=use_cache,
                Qformer_past_key_values=Qformer_past_key_values,
                cache_position=cache_position,
                gradient_checkpointing=gradient_checkpointing
                
            )

            # 5) Self-attention: Q=K=V=cross_attention_output
            self_attention_output, Qformer_past_key_values = self.self_attention(
                hidden_states=cross_attention_output,
                use_cache=use_cache,
                Qformer_past_key_values=Qformer_past_key_values,
                cache_position=cache_position,
                gradient_checkpointing=gradient_checkpointing
            )

            # 6) Add the attention output into inputs_embeds.
            inputs_embeds = inputs_embeds + ih_embed + self_attention_output * alpha.unsqueeze(-1)
        else:
            # If ih_ids is not provided, skip Qformer cross/self attention.
            # Keep inputs_embeds or embed_tokens.
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)

        # 7) Run the Llama backbone to get hidden_states.
        #    Note: Llama itself also has official past_key_values.
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,  # llama cache
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute the last logits_to_keep logits.
        # (If = 0, keep all.)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            # Backward-compatible tuple output.
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 8) Use QformerOutputWithPast and return both Llama and Qformer caches.
        return QformerOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            Qformer_past_key_values=Qformer_past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

