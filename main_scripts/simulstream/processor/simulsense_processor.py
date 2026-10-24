from main_scripts.release_utils import checkpoint_file_in, relocate_checkpoint
import json
from pathlib import Path

from simulstream.server.speech_processors.base import BaseSpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput
from transformers import DynamicCache
from types import SimpleNamespace
from typing import List
from fairseq import checkpoint_utils, tasks
from safetensors.torch import load_file
from main_scripts.models.simulsense_model import WeightPredictor
from main_scripts.models.simulsense_model import language_code_mapper
import torch
import numpy as np
import os
from copy import deepcopy
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class SimulSenseProcessor(BaseSpeechProcessor):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.dtype = self.model.dtype
        self.device = f'cuda:{getattr(config, "device", 0)}'
        self.model = self.model.to(self.dtype).to(self.device)
        self.weight_predictor = self.weight_predictor.to(self.dtype).to(self.device)
        self.tokenizer = self.model.llm_tokenizer
        self.threshold = getattr(config, "threshold")
        self.num_sense_units = 0
        self.src_lang = getattr(config, 'source_lang', None)
        self.tgt_lang = getattr(config, 'target_lang', None)
        self.stop_token_ids = [self.model.eos_token_id]
        self.decode_method = config.decode_config.decode_method
        self.decode_config = self.set_generation_config(config, self.decode_method)
        # Prompt
        prompt = self.model.prompt['st']
        prompt[0]['content'] = ''.join(prompt[0]['content'])
        prompt[1]['content'] = prompt[1]['content'].replace('<src_lang>', self.model.lang_codes[self.src_lang])
        prompt[1]['content'] = prompt[1]['content'].replace('<tgt_lang>', self.model.lang_codes[self.tgt_lang])
        prompt = self.model.llm_tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        audio_start_index = prompt.index(self.model.audio_start_id)
        base_prompt = torch.tensor(prompt[:audio_start_index]).unsqueeze(0).to(self.model.device)
        self.prompt = torch.tensor(prompt).unsqueeze(0).to(self.model.device)
        # Initialize kv cache with system prompt
        base_cache = DynamicCache()
        if self.decode_method == 'beam':
            base_prompt = base_prompt.repeat_interleave(self.decode_config.beam_num, dim=0)
        _ = self.model.llm(
            input_ids=base_prompt,
            use_cache=True,
            past_key_values=base_cache,
        )
        self.base_cache = base_cache
        self.chunk_size = getattr(config, "speech_chunk_size") # in seconds
        self.audio_max_len = (getattr(config, "audio_max_duration")-self.chunk_size) * 16000
        self.unprocessed_wav_len = 0
        self.max_wait_len = getattr(config, "max_wait_duration") * 16000 # Maximum wait length in number of samples; in case long time no translation is generated
        self.decision_trace_path = getattr(config, "decision_trace_path", None)
        if self.decision_trace_path:
            trace_path = Path(self.decision_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")
        self.utt_id = 0
        self.context_segment_id = 0
        self._reset_decision_trace_state()
    
    def set_generation_config(self, config: SimpleNamespace, decode_method) -> SimpleNamespace:
        decode_config = getattr(config, 'decode_config')
        assert decode_method in ['beam', 'sample', 'greedy'], f"Invalid decode method: {decode_method}"
        config = getattr(decode_config, decode_method)
        logger.info(f"{decode_method} decoding configuration: {config}")

        return config
        
    @classmethod
    def load_model(cls, config: SimpleNamespace):
        ckpt_path = config.ckpt_path
        checkpoint_file = checkpoint_file_in(ckpt_path)
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_file)
        combined_state_dict = OrderedDict()
        state['model'] = OrderedDict(state['model'])
        peft_adaptor_state_dict = load_file(os.path.join(ckpt_path, 'adapter_model.safetensors'), device='cpu')
        new_peft_adaptor_state_dict = {}
        for key, value in peft_adaptor_state_dict.items():
            new_key = 'llm.' + key
            if "embed_tokens.lora" in new_key:
                new_key = new_key + '.default'
            else:
                if 'lm_head.base_layer' not in new_key and 'embed_tokens.base_layer' not in new_key:
                    new_key = new_key.replace('.weight', '.default.weight')
            new_peft_adaptor_state_dict[new_key] = value
        # Combine the state dicts
        combined_state_dict.update(state['model'])
        combined_state_dict.update(new_peft_adaptor_state_dict)
        state['model'] = combined_state_dict

        relocate_checkpoint(state, config.source_lang, config.target_lang)
        task_args = state["cfg"]["task"]
        task = tasks.setup_task(task_args)
        model_cfg = state["cfg"]["model"]
        setattr(model_cfg, "inference", True)
        cls.model = task.build_model(model_cfg)
        logger.info(f"Loading model from {ckpt_path}")
        
        loading_details = cls.model.load_state_dict(state["model"], strict=False)
        if "llm.base_model.model.lm_head.default.weight" in loading_details.unexpected_keys:
            # this is due to a mistake in saving
            loading_details.unexpected_keys.remove("llm.base_model.model.lm_head.default.weight")
        if "llm.base_model.model.model.embed_tokens.default.weight" in loading_details.unexpected_keys:
            # this is due to a mistake in saving
            loading_details.unexpected_keys.remove("llm.base_model.model.model.embed_tokens.default.weight")
        if len(loading_details.unexpected_keys) > 0:
            raise ValueError(
                f"Unexpected keys in state_dict: {loading_details.unexpected_keys}"
            )
        
        # Build SimulSense weight predictor
        encoder_dim = cls.model.audio_encoder.config.d_model
        simulsense_path = getattr(config, "simulsense_path")
        weight_predictor_ckpt = 'sense_units_detector.pt'
        weight_predictor_path = os.path.join(simulsense_path, weight_predictor_ckpt)
        other_state_path = os.path.join(simulsense_path, 'other_states.pt')
        other_state = checkpoint_utils.load_checkpoint_to_cpu(other_state_path)
        simul_sense_args = other_state['cfg']['model']
        cls.weight_predictor = WeightPredictor(encoder_dim, args=simul_sense_args)
        state_dict = torch.load(weight_predictor_path, map_location='cpu')
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('weight_predictor.', '')
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        loading_details = cls.weight_predictor.load_state_dict(state_dict, strict=False)
        if len(loading_details.missing_keys) > 0:
            raise ValueError(
                f"Missing keys in weight predictor state_dict: {loading_details.missing_keys}"
            )
        cls.weight_predictor.eval()
        cls.model.eval()

    def _preprocess(self, waveform: np.float32) -> torch.Tensor:
        """
        Convert a raw audio waveform into model-ready features.
        """
        if self.audio_history is not None:
            self.audio_history = np.concatenate((self.audio_history, waveform))
        else:
            self.audio_history = waveform

        features, encoder_padding_mask = self.model.extract_features(self.audio_history) # (B, 128, 3000)
        B, T_mel = encoder_padding_mask.shape
        T_enc = T_mel // 2  # Whisper Encoder downsamples the input by 2
        downsampled_mask = encoder_padding_mask.view(features.size(0), T_enc, 2).max(dim=-1)[0]  # (B, 1500)  # Downsample the mask to match the Whisper encoder output
        actual_len = downsampled_mask.sum(dim=-1).max().item()  # (B,)  # Actual length of the features without padding
        features = features.to(self.dtype)
        encoder_padding_mask = downsampled_mask.to(self.dtype)

        encoder_states = self.model.audio_encoder(input_features=features, 
                                   attention_mask=encoder_padding_mask,
                                   return_dict=True).last_hidden_state
        encoder_states = encoder_states[:, :actual_len, :]  # (B, T_enc, C)
        return encoder_states

    def simulsense_policy(self, speech_embeds: torch.Tensor):
        """
        Decide whether to write or wait based on SimulSense weight predictor.
        """
        lang_id = [language_code_mapper.lang_to_id[self.src_lang]]
        output = self.weight_predictor(speech_embeds, lang_id)
        weights = output['weights']
        weights_quantity = float(output['quantity'].item())
        cur_num_sense_units = (weights_quantity // self.threshold)
        if cur_num_sense_units >= self.num_sense_units + 1:
            self.num_sense_units = cur_num_sense_units
            write = True
        else:
            write = False
        return write, weights_quantity
    
    def _generate(self, speech_embeds: torch.Tensor) -> List[str]:
        speech_embeds = self.model.forward_adaptor(speech_embeds)
        llm_input = self.model.wrap_prompt(speech_embeds, self.prompt) # (1, T', C)
        if self.text_history:
            prev_token_ids = self.tokenizer.convert_tokens_to_ids(self.text_history)
            prev_token_embeds = self.model.llm.model.model.embed_tokens(
                torch.tensor(prev_token_ids).unsqueeze(0).to(self.model.device)
            )
            llm_input = torch.cat([llm_input, prev_token_embeds], dim=1)
        cache = deepcopy(self.base_cache)
        hypos = self.model.llm_generate(
            llm_input, 
            decode_config=self.decode_config,
            eos_tokens=self.stop_token_ids,
            kv_cache=cache,
        )['sequences']
        token_ids = hypos[0].tolist()
        if token_ids[-1] == self.model.eos_token_id:
            token_ids = token_ids[:-1] # Remove EOS token
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)

        return tokens

    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        text = self.tokenizer.convert_tokens_to_string(generated_tokens)
        output_tokens = [text] if text else []
        return IncrementalOutput(
            new_tokens=output_tokens,
            new_string=text,
            deleted_tokens=[],
            deleted_string="",
        )
    
    def tokens_to_string(self, tokens: List[str]) -> str:
        if self.tgt_lang in {"zh", "ja", "ko"}:
            return "".join(tokens)
        return " ".join(token for token in tokens if token)
    
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        speech = self._preprocess(waveform)
        seg_finished = len(self.audio_history) >= self.audio_max_len
        wait_too_long = self.unprocessed_wav_len >= self.max_wait_len
        chunk_index = len(self.decisions)
        write, unit_score = self.simulsense_policy(speech)
        actual_write = seg_finished or wait_too_long or write

        if actual_write:
            generated_tokens = self._generate(speech)
            self._update_text_history(generated_tokens)
            self.unprocessed_wav_len = 0
        else:
            generated_tokens = []
            self.unprocessed_wav_len += len(waveform)

        self._record_decision(chunk_index, actual_write, unit_score)
        incremental_output = self._build_incremental_outputs(generated_tokens)
        if seg_finished:
            self._write_decision_trace("context_reset")
            self._reset_context_state()
            self.context_segment_id += 1
        return incremental_output

    def _record_decision(self, chunk_index: int, write: bool, unit_score: float) -> None:
        self.valid_chunk_indices.append(chunk_index)
        self.unit_scores.append(unit_score)
        self.decisions.append("W" if write else "R")
        if write:
            self.write_indices.append(chunk_index)

    def _reset_decision_trace_state(self) -> None:
        self.valid_chunk_indices = []
        self.unit_scores = []
        self.decisions = []
        self.write_indices = []

    def _write_decision_trace(self, reset_reason: str) -> None:
        if not self.decision_trace_path or len(self.decisions) == 0:
            return
        record = {
            "utt_id": self.utt_id,
            "wav_name": None,
            "context_segment_id": self.context_segment_id,
            "threshold": float(self.threshold),
            "speech_chunk_size": float(self.chunk_size),
            "num_chunks": len(self.decisions),
            "valid_chunk_indices": self.valid_chunk_indices,
            "simulsense_decisions": self.decisions,
            "write_indices": self.write_indices,
            "unit_scores": self.unit_scores,
            "reset_reason": reset_reason,
        }
        with open(self.decision_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _reset_context_state(self) -> None:
        super().clear()
        self.text_history = None
        self.audio_history = None
        self.num_sense_units = 0
        self.unprocessed_wav_len = 0
        self._reset_decision_trace_state()

    def clear(self) -> None:
        self._reset_context_state()

    def _update_text_history(self, new_tokens: List[str]) -> List[str]:
        if self.text_history:
            self.text_history += new_tokens
        else:
            self.text_history = new_tokens

    def _update_speech_history(self, audio_history: np.float32) -> None:
        pass

    def set_target_language(self, language: str) -> None:
        pass

    def set_source_language(self, language: str) -> None:
        pass

    def end_of_stream(self) -> IncrementalOutput:
        generated_tokens = []
        if len(self.decisions) > 0:
            if self.decisions[-1] != "W" and self.audio_history is not None:
                speech = self._preprocess(np.array([], dtype=np.float32))
                generated_tokens = self._generate(speech)
                self._update_text_history(generated_tokens)
                self.decisions[-1] = "W"
                last_chunk = self.valid_chunk_indices[-1]
                if last_chunk not in self.write_indices:
                    self.write_indices.append(last_chunk)
            self._write_decision_trace("end_of_stream")
        self.utt_id += 1
        self.context_segment_id = 0
        self._reset_context_state()
        return self._build_incremental_outputs(generated_tokens)