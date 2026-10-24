from main_scripts.release_utils import checkpoint_file_in, relocate_checkpoint
import sys
import os
import yaml
import torch
from types import SimpleNamespace
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from argparse import ArgumentParser
from base_agent import BaseAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import Action, WriteAction, ReadAction
from simuleval.agents import AgentPipeline
from simuleval.utils import entrypoint
from typing import Optional
import logging
from fairseq import checkpoint_utils
from collections import OrderedDict
from safetensors.torch import load_file
from main_scripts.models.simulsense_model import WeightPredictor
from transformers.cache_utils import DynamicCache
from transformers import StoppingCriteriaList
from main_scripts.models.utils import StoppingCriteriaSub
from main_scripts.models.simulsense_model import language_code_mapper
from copy import deepcopy

logger = logging.getLogger(__name__)

class SimulSenseState(AgentStates):
    def reset(self):
        super().reset()
        self.prev_num_sense_units = 0
        self.llm_input = None
        self.prev_tokens = []

class SimulSenseAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.src_lang = getattr(self.args, 'source_lang', None)
        self.tgt_lang = getattr(self.args, 'target_lang', None)
        self.stop_token_ids = [self.model.eos_token_id]
        self.decode_config = self.set_generation_config(args)
        self.decode_method = getattr(args, 'decode_method')
        self.no_repeat_ngram_size = getattr(args, 'no_repeat_ngram_size', 0)
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
        self.states.base_cache = base_cache

    def build_states(self) -> AgentStates:
        states = SimulSenseState()
        return states

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--source-lang', type=str, help='source language')
        parser.add_argument('--target-lang', type=str, help='target language')
        parser.add_argument('--device', type=str, default='0', help='device to run the model')
        parser.add_argument('--model-path', type=str, help='path to the model checkpoint')
        parser.add_argument('--simul-sense', type=str, help='path to the simulsense model checkpoint')
        parser.add_argument('--decode-method', type=str, default='greedy', help='Decoding method')
        parser.add_argument('--bf16', action='store_true', help='Use bf16 precision for the model')
        parser.add_argument('--no-repeat-ngram-size', type=int, default=0, help='n-gram size for no repeat')
        parser.add_argument('--audio-end-thred', type=float, help='Threshold to consider the acoustic context is sufficient for generation')

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states
        speech_embeds = self.model.forward_audio_encoder(states.source)

        # If the source is finished, generate the final translation directly
        if states.source_finished:
            speech_embeds = self.model.forward_adaptor(speech_embeds)
            hypos = self.generate_hypothesis(speech_embeds)
            translation = self.model.decode_fn(hypos)[0]
            return WriteAction(translation, finished=states.source_finished)
        else:
            # Predict weights for making decision
            lang_id = [language_code_mapper.lang_to_id[self.src_lang]]
            output = self.weight_predictor(speech_embeds, lang_id)
            # weights = output['weights'][0].cpu().float().numpy()  # (T,)
            weights_quantity = output['quantity'].item()
            cur_num_sense_units = (weights_quantity // self.args.audio_end_thred)
            if cur_num_sense_units >= states.prev_num_sense_units + 1:
                states.prev_num_sense_units = cur_num_sense_units
                # Project encoder states to LLM dimension
                speech_embeds = self.model.forward_adaptor(speech_embeds)
                # Generate hypothesis
                hypos = self.generate_hypothesis(speech_embeds)
                states.prev_tokens.extend(hypos[0].tolist()[: -1])  # Exclude eos token
                translation = self.model.decode_fn(hypos)[0]
                return WriteAction(translation, finished=states.source_finished)
            else:
                return ReadAction()

    def generate_hypothesis(self, speech_embeds):
        llm_input = self.model.wrap_prompt(speech_embeds, self.prompt) # (1, T', C)
        if len(self.states.prev_tokens) > 0:
            prev_token_embeds = self.model.llm.model.model.embed_tokens(
                torch.tensor(self.states.prev_tokens).unsqueeze(0).to(self.model.device)
            )
            llm_input = torch.cat([llm_input, prev_token_embeds], dim=1)
        cache = deepcopy(self.states.base_cache)
        hypos = self.model.llm_generate(
            llm_input, 
            decode_config=self.decode_config,
            eos_tokens=self.stop_token_ids,
            kv_cache=cache,
        )['sequences']
        hypos = self.truncate_repeat_ngrams(hypos)
        return hypos
    
    def truncate_repeat_ngrams(self, tokens):
        """ 
        Check and truncate tokens to avoid repeating n-grams
        tokens: tensor[1, T] 
        """
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
        # Generation
        decode_config = getattr(args, 'decode_config', 'main_scripts/config/decode_config.yaml')
        decode_config = yaml.safe_load(open(decode_config))
        decode_method = getattr(args, 'decode_method', None)
        assert decode_method in ['beam', 'sample', 'greedy'], f"Invalid decode method: {decode_method}"
        logger.info(f"Decoding method: {decode_method}")
        if decode_method == 'beam':
            decode_config = decode_config['beam_search']
        elif decode_method == 'sample':
            decode_config = decode_config['sample']
        elif decode_method == 'greedy':
            decode_config = None
            return decode_config
        else:
            raise ValueError(f"Invalid decoding method: {decode_method}")
        logger.info(f"{decode_method} decoding configuration: {decode_config}")
        decode_config = SimpleNamespace(**decode_config)

        return decode_config

    def load_weight_predictor(self, encoder_dim, args):
        simul_sense_path = getattr(args, 'simul_sense', None)
        if not os.path.exists(simul_sense_path):
            raise FileNotFoundError(f"SimulSense Model checkpoint not found at {simul_sense_path}")
        logger.info(f"Loading SimulSense weight predictor from {simul_sense_path}")
        weight_predictor_ckpt = 'sense_units_detector.pt'
        other_state_ckpt = 'other_states.pt'
        weight_predictor_path = os.path.join(simul_sense_path, weight_predictor_ckpt)
        other_state_path = os.path.join(simul_sense_path, other_state_ckpt)
        other_state = checkpoint_utils.load_checkpoint_to_cpu(other_state_path)
        simul_sense_args = other_state['cfg']['model']

        weight_predictor = WeightPredictor(encoder_dim, args=simul_sense_args)
        weight_predictor_states = torch.load(weight_predictor_path, map_location='cpu')
        new_states = OrderedDict()
        for key in weight_predictor_states.keys():
            new_key = key.replace('weight_predictor.', '') if key.startswith('weight_predictor.') else key
            new_states[new_key] = weight_predictor_states[key]
        weight_predictor_states = new_states
        loading_details = weight_predictor.load_state_dict(weight_predictor_states, strict=False)
        if len(loading_details.missing_keys) > 0:
            raise ValueError(
                f"Missing keys in weight predictor state_dict: {loading_details.missing_keys}"
            )

        return weight_predictor

    def load_model_vocab(self, args):
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        checkpoint_file = checkpoint_file_in(model_path)
        is_peft = os.path.isdir(model_path)
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_file)
        
        combined_state_dict = OrderedDict()
        state['model'] = OrderedDict(state['model'])
        peft_adaptor_state_dict = load_file(os.path.join(model_path, 'adapter_model.safetensors'), device='cpu')
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

        relocate_checkpoint(state, args.source_lang, args.target_lang)
        task_args = state["cfg"]["task"]
        task = self.set_up_task(task_args)
        model_cfg = state["cfg"]["model"]
        setattr(model_cfg, "inference", True)
        self.model = task.build_model(model_cfg)
        logger.info(f"Loading model from {model_path}")
        
        loading_details = self.model.load_state_dict(state["model"], strict=False)
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
        encoder_dim = self.model.audio_encoder.config.d_model
        self.weight_predictor = self.load_weight_predictor(encoder_dim, args)

        self.weight_predictor.eval()
        self.model.eval()
        self.weight_predictor.to(self.device)
        if getattr(args, "fp16", False):
            self.model.half()
            self.weight_predictor.half()
            logger.info("fp16 for inference is enabled.")
        elif getattr(args, "bf16", False):
            self.weight_predictor.bfloat16()
            self.model.bfloat16()
            logger.info("bf16 for inference is enabled.")
        else:
            logger.info("Using fp32 for inference.")

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.states.dict = self.dict['tgt']
            