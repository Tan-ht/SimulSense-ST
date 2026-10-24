from main_scripts.release_utils import checkpoint_file_in, relocate_checkpoint
import logging
import os
import sys
import yaml
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from typing import Optional

import torch
from fairseq import checkpoint_utils
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent import BaseAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import WriteAction

logger = logging.getLogger(__name__)


class STPolicyState(AgentStates):
    def reset(self):
        super().reset()
        self.prev_tokens = []


class STPolicyAgentBase(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.src_lang = getattr(self.args, "source_lang", None)
        self.tgt_lang = getattr(self.args, "target_lang", None)
        self.stop_token_ids = [self.model.eos_token_id]
        self.decode_config = self.set_generation_config(args)
        self.decode_method = getattr(args, "decode_method")
        self.no_repeat_ngram_size = getattr(args, "no_repeat_ngram_size", 0)
        self._prepare_prompt()

    def build_states(self) -> AgentStates:
        return STPolicyState()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "base_cache"):
            self.states.base_cache = self.base_cache

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--source-lang", type=str, help="source language")
        parser.add_argument("--target-lang", type=str, help="target language")
        parser.add_argument("--device", type=str, default="0", help="device to run the model")
        parser.add_argument("--model-path", type=str, help="path to the model checkpoint")
        parser.add_argument("--decode-method", type=str, default="greedy", help="Decoding method")
        parser.add_argument("--bf16", action="store_true", help="Use bf16 precision for the model")
        parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="n-gram size for no repeat")

    def _prepare_prompt(self):
        prompt = self.model.prompt["st"]
        prompt[0]["content"] = "".join(prompt[0]["content"])
        prompt[1]["content"] = prompt[1]["content"].replace(
            "<src_lang>", self.model.lang_codes[self.src_lang]
        )
        prompt[1]["content"] = prompt[1]["content"].replace(
            "<tgt_lang>", self.model.lang_codes[self.tgt_lang]
        )
        prompt = self.model.llm_tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        audio_start_index = prompt.index(self.model.audio_start_id)
        base_prompt = torch.tensor(prompt[:audio_start_index]).unsqueeze(0).to(self.model.device)
        self.prompt = torch.tensor(prompt).unsqueeze(0).to(self.model.device)

        base_cache = DynamicCache()
        if self.decode_method == "beam":
            base_prompt = base_prompt.repeat_interleave(self.decode_config.beam_num, dim=0)
        _ = self.model.llm(
            input_ids=base_prompt,
            use_cache=True,
            past_key_values=base_cache,
        )
        self.base_cache = base_cache
        self.states.base_cache = self.base_cache

    def write_translation(self, states: AgentStates) -> WriteAction:
        speech_embeds = self.model.forward_audio_encoder(states.source)
        speech_embeds = self.model.forward_adaptor(speech_embeds)
        hypos = self.generate_hypothesis(speech_embeds, states)
        states.prev_tokens.extend(hypos[0].tolist()[:-1])
        translation = self.model.decode_fn(hypos)[0]
        return WriteAction(translation, finished=states.source_finished)

    def generate_hypothesis(self, speech_embeds, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states
        llm_input = self.model.wrap_prompt(speech_embeds, self.prompt)
        if len(states.prev_tokens) > 0:
            prev_token_embeds = self.model.llm.model.model.embed_tokens(
                torch.tensor(states.prev_tokens).unsqueeze(0).to(self.model.device)
            )
            llm_input = torch.cat([llm_input, prev_token_embeds], dim=1)
        cache = deepcopy(states.base_cache)
        hypos = self.model.llm_generate(
            llm_input,
            decode_config=self.decode_config,
            eos_tokens=self.stop_token_ids,
            kv_cache=cache,
        )["sequences"]
        return self.truncate_repeat_ngrams(hypos)

    def truncate_repeat_ngrams(self, tokens):
        min_ngram_size = self.no_repeat_ngram_size
        if min_ngram_size <= 0:
            return tokens

        tokens_list = tokens[0].tolist()
        length = len(tokens_list)
        for i in range(length - min_ngram_size + 1):
            ngram = tokens_list[i : i + min_ngram_size]
            for j in range(i + min_ngram_size, length - min_ngram_size + 1):
                if tokens_list[j : j + min_ngram_size] == ngram:
                    return tokens[:, : i + min_ngram_size]
        return tokens

    def set_generation_config(self, args):
        decode_config = getattr(args, "decode_config", "main_scripts/config/decode_config.yaml")
        decode_config = yaml.safe_load(open(decode_config))
        decode_method = getattr(args, "decode_method", None)
        assert decode_method in ["beam", "sample", "greedy"], f"Invalid decode method: {decode_method}"
        logger.info(f"Decoding method: {decode_method}")
        if decode_method == "beam":
            decode_config = decode_config["beam_search"]
        elif decode_method == "sample":
            decode_config = decode_config["sample"]
        elif decode_method == "greedy":
            return None
        else:
            raise ValueError(f"Invalid decoding method: {decode_method}")
        logger.info(f"{decode_method} decoding configuration: {decode_config}")
        return SimpleNamespace(**decode_config)

    def load_model_vocab(self, args):
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        checkpoint_file = checkpoint_file_in(model_path)
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_file)
        combined_state_dict = OrderedDict()
        state["model"] = OrderedDict(state["model"])
        peft_adaptor_path = os.path.join(model_path, "adapter_model.safetensors")
        peft_adaptor_state_dict = load_file(peft_adaptor_path, device="cpu")
        new_peft_adaptor_state_dict = {}
        for key, value in peft_adaptor_state_dict.items():
            new_key = "llm." + key
            if "embed_tokens.lora" in new_key:
                new_key = new_key + ".default"
            elif "lm_head.base_layer" not in new_key and "embed_tokens.base_layer" not in new_key:
                new_key = new_key.replace(".weight", ".default.weight")
            new_peft_adaptor_state_dict[new_key] = value

        combined_state_dict.update(state["model"])
        combined_state_dict.update(new_peft_adaptor_state_dict)
        state["model"] = combined_state_dict

        relocate_checkpoint(state, args.source_lang, args.target_lang)
        task_args = state["cfg"]["task"]
        task = self.set_up_task(task_args)
        model_cfg = state["cfg"]["model"]
        setattr(model_cfg, "inference", True)
        self.model = task.build_model(model_cfg)
        logger.info(f"Loading model from {model_path}")

        loading_details = self.model.load_state_dict(state["model"], strict=False)
        if "llm.base_model.model.lm_head.default.weight" in loading_details.unexpected_keys:
            loading_details.unexpected_keys.remove("llm.base_model.model.lm_head.default.weight")
        if "llm.base_model.model.model.embed_tokens.default.weight" in loading_details.unexpected_keys:
            loading_details.unexpected_keys.remove(
                "llm.base_model.model.model.embed_tokens.default.weight"
            )
        if len(loading_details.unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys in state_dict: {loading_details.unexpected_keys}")

        self.model.eval()
        if getattr(args, "fp16", False):
            self.model.half()
            logger.info("fp16 for inference is enabled.")
        elif getattr(args, "bf16", False):
            self.model.bfloat16()
            logger.info("bf16 for inference is enabled.")
        else:
            logger.info("Using fp32 for inference.")

        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.states.dict = self.dict["tgt"]
