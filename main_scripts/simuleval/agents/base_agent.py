from main_scripts.release_utils import checkpoint_file_in, relocate_checkpoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import Action
from typing import Optional
from argparse import ArgumentParser
from transformers import AutoProcessor
from fairseq import checkpoint_utils, tasks
from collections import OrderedDict
import main_scripts.models
from safetensors.torch import load_file
import torch
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from transformers import DynamicCache

logger = logging.getLogger(__name__)


class SpeechState(AgentStates):
    def reset(self):
        super().reset()
        self.prev_tokens = None
        self.prev_audio_len = 0
        self.llm_kv_cache = DynamicCache()

class BaseAgent(SpeechToTextAgent):
    def __init__(self, args):
        super().__init__(args)
        self.device = 'cuda:'+self.args.device if torch.cuda.is_available() else 'cpu'
        self.load_model_vocab(args)
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            self.model.to(self.device)
        self.eos_id = self.model.llm_tokenizer.eos_token_id
        
        torch.set_grad_enabled(False)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--device', type=str, default='0', help='device to run the model')
        parser.add_argument('--model-path', type=str, help='path to the model checkpoint')
        parser.add_argument('--no-repeat-ngram-size', type=int, default=0, help='n-gram size for no repeat')

    def to_device(self, tensor):
        return tensor.to(self.device)

    def _ensure_state_attributes(self, states: AgentStates):
        if not hasattr(states, "last_translated"):
            states.last_translated = 0
        if not hasattr(states, "stable_hypothesis"):
            states.stable_hypothesis = []
        if not hasattr(states, "hypothesis"):
            states.hypothesis = []
        if not hasattr(states, "prev_tokens"):
            states.prev_tokens = None
        if not hasattr(states, "clips_num"):
            states.clips_num = 0
        if not hasattr(states, "audio_start"):
            states.audio_start = None
        if not hasattr(states, "segment_infos"):
            states.segment_infos = []
            
    def set_up_task(self, task_args):
        return tasks.setup_task(task_args)
    
    def load_model_vocab(self, args):
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        checkpoint_file = checkpoint_file_in(model_path)
        is_peft = os.path.isdir(model_path)
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_file)
        
        if is_peft:
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
        if len(loading_details.unexpected_keys) > 0:
            raise ValueError(
                f"Unexpected keys in state_dict: {loading_details.unexpected_keys}"
            )
        self.model.eval()
        if getattr(args, "fp16", False):
            self.model.half()
            logger.info("fp16 for inference is enabled.")
        elif getattr(args, "bf16", False):
            self.model.bfloat16()
            logger.info("bf16 for inference is enabled.")
        else:
            logger.info("Using fp32 for inference.")

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.states.dict = self.dict['tgt']

    def build_states(self) -> AgentStates:
        states = SpeechState()

        return states

    def _get_prefix(self, states):
        return states.prev_tokens
        
    def _get_prefix_len(self, prefix):
        return len(prefix[0]) if prefix is not None else 0
    
    def decode_fn(self, tokens):
        return self.model.llm_tokenizer.batch_decode(tokens, skip_special_tokens=False)

    def generate_hypothesis(self, states, args):
        """
        This method takes *states* and *prefix_tokens* as inputs and generates and returns the mostly likely
        translation hypothesis *hypo*.
        """
        output = self.model.generate(states.source, states.prev_tokens, args=args)
        
        return output

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        pass

    def plot_attn(self, data, x_label=None, y_label=None, save_path=None, font_size=14, show_cbar=True):
        '''
        Plot the attention model heatmap
        Args:
            data: attn_matrix with shape [ty, tx], cutted before 'PAD'
            X_label: list of size tx, encoder tags
            Y_label: list of size ty, decoder tags
        '''
        if x_label is None:
            x_label = [str(i) for i in range(data.shape[1])]
        if y_label is None:
            y_label = [str(i) for i in range(data.shape[0])]
        if save_path is None:
            save_path = 'attn_heatmap.png'

        y_label = y_label[::-1]
        if type(data) is not np.ndarray:
            data = data.cpu().numpy()
        data = data[::-1, :]
        fig, ax = plt.subplots(figsize=(20, 8))  # set figure size
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
        ax.set_ylim(0, data.shape[0])

        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        
        # Set axis labels
        if x_label is not None and y_label is not None:
            xticks = [x + 0.5 for x in range(len(x_label)) if x % 5 == 0]
            xtick_labels = [x_label[x] for x in range(len(x_label)) if x % 5 == 0]

            # Ensure the last label is included
            if len(x_label) % 5 != 0 or xticks[-1] != len(x_label) - 0.5:
                xticks.append(len(x_label) - 0.5)
                xtick_labels.append(x_label[-1])

            ax.set_xticks(xticks, minor=False)  # major ticks
            ax.set_xticklabels(xtick_labels, minor=False, rotation=0, ha='center', fontsize=font_size)  # labels should be 'unicode'

            yticks = [y + 0.5 for y in range(len(y_label))]
            ax.set_yticks(yticks, minor=False)
            ax.set_yticklabels(y_label, minor=False, va='center', fontsize=font_size)  # labels should be 'unicode'
            
        # Add color bar
        if show_cbar:
            cbar = plt.colorbar(heatmap)
        # Save Figure
        print(f"Saving figures {save_path}")
        fig.savefig(save_path)  # save the figure to file
        plt.close(fig)  # close the figure




