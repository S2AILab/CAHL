import os
import json
import torch
import random
# import logging
import inspect
import warnings
import transformers
import torch.nn as nn
import torch.distributed as dist

from datasets import Dataset
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Sequence, Dict
from trl.trainer.utils import ConstantLengthDataset
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.utils import (
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateOutput
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    StaticCache
)
from transformers.models.llama.modeling_llama import KwargsForCausalLM, LlamaConfig, LlamaModel
from transformers.utils import is_torchdynamo_compiling, logging


logger = logging.get_logger(__name__)

IGNORE_INDEX = -100


class LlamaIHConfig(LlamaConfig):
    model_type = 'llama_ih'


class LlamaIHModel(LlamaModel):
    config_class = LlamaIHConfig

    def __init__(self, config: LlamaConfig):
        super(LlamaIHModel, self).__init__(config)



class ConstantLengthDatasetForInstructionHisrarchy(ConstantLengthDataset):
    """
    When dataset is pretokenized, 
        ConstantLengthDataset only take `input_ids` using a `formatting_func` which returns `x['input_ids']`
        ConstantLengthDatasetForInstructionHisrarchy will take `input_ids` and `ih_ids` using a `formatting_func` which returns `x` directly
    Although the input datasets are in different format, it is compatible that they yield the same output format.
    """
    def __init__(
        self,
        tokenizer,
        dataset,
        ih_ids=4,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        super().__init__(
            tokenizer,
            dataset,
            dataset_text_field,
            formatting_func,
            infinite,
            seq_length,
            num_of_sequences,
            chars_per_token,
            eos_token_id,
            shuffle,
            append_concat_token,
            add_special_tokens
        )
        self.ih_ids = ih_ids

    def __iter__(self):
        # ConstantLengthDatasetForInstructionHisrarchy takes a `formatting_func` -> {'input_ids': ..., 'ih_ids': ...}
        #   Just keep it to align to the behavior of `ConstantLengthDataset`, although it can be removed.
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    x = self.formatting_func(next(iterator))
                    buffer.append({
                        'input_ids': x['input_ids'],
                        'ih_ids': x['ih_ids']
                    })
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            if self.shuffle:
                random.shuffle(buffer)
            if self.pretokenized:
                print("======== directly use buffer as tokenized_inputs")
                tokenized_inputs = buffer
            else:
                # non-pretokenized dataset takes format: {'instruction': text, 'input': text, 'output': text}
                print("======== tokenizing buffer as tokenized_inputs")
                tokenized_inputs = self._tokenize(buffer)
            all_token_ids = []
            all_ih_ids = []
            # for tokenized_input in tokenized_inputs:
            for i in range(len(tokenized_inputs)):
                if self.append_concat_token:    # todo: support special token ih_id
                    tokenized_inputs[i]['input_ids'] += [self.concat_token_id]
                    tokenized_inputs[i]['ih_ids'] += [2]    
                all_token_ids.extend(tokenized_inputs[i]['input_ids'])
                all_ih_ids.extend(tokenized_inputs[i]['ih_ids'])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                ih_ids = all_ih_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length and len(ih_ids) == self.seq_length:
                    examples.append({
                        'input_ids': input_ids,
                        'ih_ids': ih_ids
                    })
            if self.shuffle:
                # Shuffle again, otherwise split examples occur in consecutive tensors.
                random.shuffle(examples)
            # for example in examples:
            for i in range(len(examples)):
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(examples[i]['input_ids']),
                    "labels": torch.LongTensor(examples[i]['input_ids']),
                    "ih_ids": torch.LongTensor(examples[i]['ih_ids'])
                }
    
    def _tokenize(self, buffer):
        system, insts, inputs, texts = [], [], []
        for x in buffer:
            if 'system' in x:
                system.append(x['system'])
            else:
                system.append('')
            insts.append(x['instruction'])
            inputs.append(x['input'])
            texts.append(x['instruction'] + x['input'] + x['output'])
        sys_t = self.tokenizer(
            system,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )
        inst_t = self.tokenizer(
            insts,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )
        input_t = self.tokenizer(
            inputs,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )
        texts_t = self.tokenizer(
            texts,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=True,
            padding=False,
            truncation=False
        )
        input_ids = texts_t.input_ids
        for i in range(len(buffer)):
            inst_start_idx = len(sys_t['input_ids'][i]) + 1
            input_start_idx = inst_start_idx + len(inst_t['input_ids'][i])
            output_start_idx = input_start_idx + len(input_t['input_ids'][i])
            ih_id = torch.zeros_like(input_ids[i], dtype=torch.long)    # 0: system
            ih_id[inst_start_idx: input_start_idx] = 1 if self.ih_size == 4 else 0  # 0 & 1: instruction
            ih_id[input_start_idx: output_start_idx] = 2 if self.ih_size == 4 else 1    # 1 & 2: data
            ih_id[output_start_idx: ] = 3 if self.ih_size == 4 else 2   # 2 & 3: assistant
            buffer[i] = {
                "input_ids": input_ids[i],
                "ih_ids": ih_id
            }
        
        return buffer


class GenerationMixinForIH(GenerationMixin):
    def __init__(self, ih_size=3):
        super().__init__()
        self.ih_size = ih_size
        self.transformer = None

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """
        # print("====== ih_ids:", kwargs['ih_ids'])
        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # 2. Generic cache-dependent input preparation
        ih_ids = kwargs.pop('ih_ids', None)
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
                ih_ids = ih_ids[:, -cache_position.shape[0] :] if ih_ids is not None else None
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
                # batch_size = input_ids.shape[0]
                # In completion mode, all new tokens are in the `assistant` level, here are `ih_ids=2`
                if ih_ids is not None:
                    ih_ids = ih_ids[:, -cache_position.shape[0] :]
            assert ih_ids.shape[1] == input_ids.shape[1], "ih_ids.shape[1] must match input_ids.shape[1]."

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs["inputs_embeds"] is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask if we are using a `StaticCache`
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 6.5. Forward ih_ids to the model
        if ih_ids is not None:
            ih_ids = ih_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["ih_ids"] = ih_ids

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), StaticCache):
            if self.device.type == "cuda":
                logger.warning_once("Using `torch.compile`.")
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device # 4.51.3, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # Keep ih_ids in sync with newly generated tokens.
            if "ih_ids" in model_kwargs and model_kwargs["ih_ids"] is not None:
                ih_ids_old = model_kwargs["ih_ids"]
                ih_ids_new = torch.full(
                    (ih_ids_old.shape[0], 1),
                    fill_value=(self.ih_size - 1),
                    dtype=ih_ids_old.dtype,
                    device=ih_ids_old.device,
                )
                model_kwargs["ih_ids"] = torch.cat([ih_ids_old, ih_ids_new], dim=-1)
            assert input_ids.shape[1] == model_kwargs["ih_ids"].shape[1], "input_ids.shape[1] must match ih_ids.shape[1]."
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )

    #     # Convert to legacy cache format if requested
    #     if (
    #         generation_config.return_legacy_cache is not False  # Should check for `True` after v4.47
    #         and not is_torchdynamo_compiling()
    #         and hasattr(result, "past_key_values")
    #         and hasattr(result.past_key_values, "to_legacy_cache")
    #         and result.past_key_values.to_legacy_cache is not None
    #     ):
    #         # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
    #         should_convert_cache = generation_config.return_legacy_cache
    #         is_user_defined_cache = user_defined_cache is not None
    #         is_default_cache_type = (
    #             type(result.past_key_values) == DynamicCache  # noqa E721
    #             or (
    #                 isinstance(result.past_key_values, EncoderDecoderCache)
    #                 and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
    #                 and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
    #             )
    #         )
    #         if not is_user_defined_cache and is_default_cache_type:
    #             logger.warning_once(
    #                 "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
    #                 "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
    #                 "keep returning the legacy format, please set `return_legacy_cache=True`."
    #             )
    #             should_convert_cache = True
    #         if should_convert_cache:
    #             result.past_key_values = result.past_key_values.to_legacy_cache()
    #     return result
