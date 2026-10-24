import os
import sys
from fairseq.models import register_model, register_model_architecture, BaseFairseqModel
import logging
from types import SimpleNamespace
import json
import copy
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, StoppingCriteriaList, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from .utils import StoppingCriteriaSub, preferred_attn_implementation

logger = logging.getLogger(__name__)

class SpecialTokens(str, Enum):
    begin_target = "<|begintarget|>"
    audio_start = "<|audio_start|>"
    audio_end = "<|audio_end|>"
    audio = "<|audio_here|>"
    asr_token = "<|ASR|>"
    st_token = "<|ST|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

@register_model("llm_sst")
class LlmSST(BaseFairseqModel):
    def __init__(self, model_components, args):
        super().__init__()
        model_components = SimpleNamespace(**model_components)
        self.num_updates = 0
        self.args = args
        self.model_task = getattr(args, 'model_task', 'st')  # Speech-to-text only

        # Languages
        self.lang_codes = {'en': 'English', 'de': 'German', 'zh': 'Chinese', 'ja': 'Japanese'}
        self.src_lang = getattr(args, 'src_lang', None)
        self.tgt_lang = getattr(args, 'tgt_lang', None)
        assert self.src_lang is not None and self.tgt_lang is not None, "Source and target languages must be specified."

        from main_scripts.release_utils import validate_pair
        validate_pair(self.src_lang, self.tgt_lang)
        if self.model_task != 'st':
            raise ValueError('This release supports speech-to-text only')

        # Model components
        self.feature_extractor = model_components.whisper_components['processor']
        self.audio_encoder = model_components.whisper_components['encoder']
        if getattr(args, 'build_whisper_decoder', False):
            self.audio_decoder = model_components.whisper_components['decoder']
        self.llm = model_components.llm
        self.llm_tokenizer = model_components.llm_tokenizer
        self.ln_speech = model_components.ln_speech
        self.llm_proj = model_components.llm_proj
        # Prepare prompt for LLM
        with open(args.prompt_path, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)
        self.prompt = raw_prompts

        # Special token ids
        self.audio_end_id = self.llm_tokenizer.convert_tokens_to_ids('<|audio_end|>')
        self.audio_start_id = self.llm_tokenizer.convert_tokens_to_ids('<|audio_start|>')
        self.audio_here_id = self.llm_tokenizer.convert_tokens_to_ids('<|audio_here|>')
        self.eos_token_id = self.llm_tokenizer.eos_token_id

        # Decode config
        decode_config_path = getattr(args, 'decode_config', None)
        with open(decode_config_path, 'r', encoding='utf-8') as f:
            decode_config_dict = yaml.safe_load(f)
        self.decode_config = SimpleNamespace(**decode_config_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model-task', type=str, choices=['st'], help='Speech-to-text translation')
        parser.add_argument('--src-lang', type=str, help='Source language code')
        parser.add_argument('--tgt-lang', type=str, help='Target language code')
        # Encoder
        parser.add_argument('--whisper-path', type=str, help='Path to the Whisper model')
        parser.add_argument('--freeze-encoder', action='store_true', help='Freeze the encoder')
        parser.add_argument('--build-whisper-decoder', action='store_true', help='Build Whisper decoder for ASR inference')
        # Adaptor
        parser.add_argument('--num-projection-layers', type=int, help='Number of layers in the projection MLP')
        # LLM
        parser.add_argument('--llm-path', type=str, help='Path to the LLM model')
        parser.add_argument('--prompt-path', type=str, default='', help='Path to the prompt file')
        # LoRA
        parser.add_argument('--lora-rank', type=int, help='Rank of the Lora matrix')
        parser.add_argument('--lora-alpha', type=int, help='Alpha of the Lora matrix')
        parser.add_argument('--lora-dropout', type=float, help='Dropout rate for Lora')
        # Generation
        parser.add_argument('--decode-config', type=str, default='main_scripts/config/decode_config.yaml', help='Path to the decoding configuration')
        parser.add_argument('--decode-method', type=str, help='Decoding method')

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    @property
    def dtype(self):
        return list(self.parameters())[0].dtype
    
    def freeze_adaptor(self):
        for param in self.llm_proj.parameters():
            param.requires_grad = False

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def cal_num_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @classmethod
    def build_audio_encoder(cls, args):
        ''' Build the audio encoder '''
        whisper_components = {}
        model_path = getattr(args, 'whisper_path', None)
        assert model_path is not None, "Whisper model path must be specified."
        if getattr(args, 'fp16', False):
            torch_dtype = torch.float16
        elif getattr(args, 'bf16', False):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        WhisperProcessor = AutoProcessor.from_pretrained(model_path)
        whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, low_cpu_mem_usage=True, attn_implementation=preferred_attn_implementation()
            )
        whisper.to(torch_dtype)
        encoder = whisper.get_encoder()
        logger.info(f"Loaded audio encoder from: {model_path}")
        if getattr(args, 'freeze_encoder', False):
            encoder._freeze_parameters()
            assert not any(p.requires_grad for p in encoder.parameters()), "Encoder parameters are not frozen."
            logger.info("Freezing audio encoder")
        whisper_components['encoder'] = encoder
        whisper_components['processor'] = WhisperProcessor
        if getattr(args, 'build_whisper_decoder', False):
            whisper_components['decoder'] = whisper.get_decoder() # Build decoder only in inference for ASR

        return whisper_components
    
    

    @classmethod
    def build_llm(cls, args):
        llm_path = getattr(args, 'llm_path', None)
        llm_name = llm_path.split("/")[-1]
        assert llm_path is not None, "LLM path must be specified."
        """ Tokenizer """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, additional_special_tokens=SpecialTokens.list())
        llm_tokenizer.padding_side = 'right'
        vocab = llm_tokenizer.get_vocab()
        
        # Update vocab file if necessary
        def save_vocab(vocab, vocab_path):
            with open(vocab_path, "w", encoding="utf-8") as f:
                for token, token_id in tqdm(sorted(vocab.items(), key=lambda x: x[1])):  # Sort by ID
                    f.write(f"{token} {token_id}\n")
        vocab_path = os.path.join(args.user_dir, 'llm_vocab/', 'Qwen3-8B-vocab.txt')
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_size = len(f.readlines())
        if vocab_size != len(vocab):
            logger.warning(f"Vocab size mismatch: {vocab_size} != {len(vocab)}. Updating {llm_name}-vocab.txt. Please run again.")
            save_vocab(vocab, vocab_path)
            sys.exit(0)
                    
        ''' Model '''
        # Set low_cpu_mem_usage=True to reduce maximum memory usage during initialization. 
        # During inference, set device_map="auto" can distribute model to multiple GPUs; model.hf_device_map to check how it distributed.
        llm_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
        llm_model.resize_token_embeddings(len(vocab))  # Expand the embedding matrix to match the tokenizer size
        logger.info(f"{llm_name} model loaded from: {llm_path}")

        # Lora
        if getattr(args, 'inference', False):
            logger.info("Building PEFT LLM with inference mode.")
            inference=True
        else:
            inference=False
        lora_rank = args.lora_rank
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=inference,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        llm_model = get_peft_model(llm_model, lora_config)
        llm_model.print_trainable_parameters()

        return llm_model, llm_tokenizer

    @classmethod
    def build_model(cls, args, task=None):
        model_components = {}
        # Build audio encoder
        whisper_components = cls.build_audio_encoder(args)
        encoder_out_dim = whisper_components['encoder'].config.d_model

        # Build the PEFT LLM model and tokenizer
        llm, llm_tokenizer = cls.build_llm(args)  
        llm_hidden_size = llm.config.hidden_size

        # Build MLP Adaptor
        linear_modules = [nn.Linear(encoder_out_dim, llm_hidden_size, bias=False)]
        for _ in range(1, args.num_projection_layers):
            linear_modules.append(nn.GELU())
            linear_modules.append(nn.Linear(llm_hidden_size, llm_hidden_size, bias=False))
        llm_proj = nn.Sequential(*linear_modules)

        model_components['whisper_components'] = whisper_components
        model_components['ln_speech'] = nn.LayerNorm(encoder_out_dim)
        model_components['llm_proj'] = llm_proj
        model_components['llm'] = llm
        model_components['llm_tokenizer'] = llm_tokenizer

        return cls(model_components, args)

    def wrap_prompt(self, speech_embeds, tokenized_prompts):
        ''' Wrap the speech embeddings with the LLM prompt. '''
        prompt_embeds = self.llm.model.model.embed_tokens(tokenized_prompts)  # B, T, C
        audio_tag_id = self.llm_tokenizer.convert_tokens_to_ids('<|audio_here|>')
        audio_idx = (tokenized_prompts[0] == audio_tag_id).nonzero(as_tuple=True)[0].item() # postion of <|audio_here|> in the prompt
        before = prompt_embeds[:, :audio_idx, :]  
        after = prompt_embeds[:, audio_idx + 1:, :]  
        wrapped_embeds = torch.cat([before, speech_embeds, after], dim=1)  

        return wrapped_embeds
    
    
    def extract_features(self, waveform, sample_rate=16000):
        processor_out = self.feature_extractor.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_attention_mask=True,
            do_normalize=True,
            device='cpu',
            return_tensors='pt',
        )
        features = processor_out['input_features']  # numpy
        features = features.to(self.device)
        pad_mask = processor_out['attention_mask']
        pad_mask = pad_mask.to(self.device)
        
        return features, pad_mask

    def forward_audio_encoder(self, waveform):
        '''
        Args:
            waveform: List[np.array], each array is a waveform
        Return:
            encoder_states: B, T, C
        '''
        mel_features, pad_mask = self.extract_features(waveform)  # (B, 128, 3000) and (B, 3000)
        B, T_mel = pad_mask.shape
        T_enc = T_mel // 2  # Whisper Encoder downsamples the input by 2
        downsampled_mask = pad_mask.view(mel_features.size(0), T_enc, 2).max(dim=-1)[0]  # (B, 1500)  # Downsample the mask to match the Whisper encoder output
        actual_len = downsampled_mask.sum(dim=-1).max().item()  # (B,)  # Actual length of the features without padding

        if getattr(self.args, 'fp16', False):
            mel_features = mel_features.half()
        elif getattr(self.args, 'bf16', False):
            mel_features = mel_features.bfloat16()

        encoder_states = self.audio_encoder(input_features=mel_features, 
                                   attention_mask=pad_mask,
                                   return_dict=True).last_hidden_state

        return encoder_states[:, :actual_len, :]  # B, T, C
    
    def forward_adaptor(self, encoder_states):
        encoder_states = self.ln_speech(encoder_states)  # layernorm to the AuT out
        encoder_states = self.llm_proj(encoder_states) # Project to LLM hidden size
        return encoder_states
    
    def forward_st(self, batch_prompts, speech_embeds, target_texts, src_langs, tgt_langs, bsz):
        batch_targets = []
        for i_, prompt in enumerate(batch_prompts):
            prompt[0]['content'] = ''.join(prompt[0]['content'])
            prompt[1]['content'] = prompt[1]['content'].replace("<src_lang>", self.lang_codes[src_langs[i_]])
            prompt[1]['content'] = prompt[1]['content'].replace("<tgt_lang>", self.lang_codes[tgt_langs[i_]])
            batch_targets.append(f"{target_texts[i_]+self.llm_tokenizer.eos_token}")
        batch_prompts = self.llm_tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokenized_prompts = self.llm_tokenizer(batch_prompts, return_tensors='pt').input_ids.to(self.device)  # B, T

        wrapped_embeds = self.wrap_prompt(speech_embeds, tokenized_prompts)  # Wrap speech embeds with prompt embeds to get the LLM input
        tokenized_target = self.llm_tokenizer(batch_targets, return_tensors='pt', padding=True).input_ids.to(self.device)  # B, T
        tokenized_target_embeds = self.llm.model.model.embed_tokens(tokenized_target)  # Get the embeddings for the target text
        llm_input = torch.cat([wrapped_embeds, tokenized_target_embeds], dim=1)

        empty_targets = (torch.ones([bsz, wrapped_embeds.size(1)], dtype=torch.long).to(self.device).fill_(-100))
        llm_target = torch.cat([empty_targets, tokenized_target], dim=1)  
        llm_target = llm_target.masked_fill(llm_target == self.llm_tokenizer.pad_token_id, -100)

        return llm_input, llm_target

    
    def forward(self, src_tokens, src_texts=None, target_texts=None, tgt_tokens=None, **kwargs):
        bsz = len(src_tokens)
        src_langs = [self.src_lang for _ in range(bsz)]
        tgt_langs = [self.tgt_lang for _ in range(bsz)]
        # Forward audio encoder
        speech_embeds = self.forward_audio_encoder(src_tokens)

        # Forward adaptor
        speech_embeds = self.forward_adaptor(speech_embeds)  
        
        batch_prompts = [copy.deepcopy(self.prompt['st']) for _ in range(bsz)]  # Create a batch of prompts for the LLM
        if self.model_task == 'st':
            output_dict = {}
            llm_input, llm_target = self.forward_st(batch_prompts, speech_embeds, target_texts, src_langs, tgt_langs, bsz)
            outputs = self.llm(inputs_embeds=llm_input, labels=llm_target)
            output_dict['st_loss'] = outputs.loss
        else:
            raise ValueError(f"Invalid model task: {self.model_task}")

        return output_dict
        
    def decode_fn(self, tokens):
        return self.llm_tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def llm_generate(self, llm_input, decode_config=None, eos_tokens=None, min_len=None, stopping=None, use_cache=True, kv_cache=None, attn_mask=None, args=None):
        """ LLM generation function """
        max_new_tokens = getattr(decode_config, 'max_new_tokens', 128)
        do_sample = getattr(decode_config, 'do_sample', False)
        num_beams = getattr(decode_config, 'beam_num', 1)
        top_p = getattr(decode_config, 'top_p', None)
        top_k = getattr(decode_config, 'top_k', None)
        temperature = getattr(decode_config, 'temperature', 0.0) # Default greedy decoding
        if do_sample:
            seed = getattr(decode_config, 'seed', 42)
            torch.manual_seed(seed)
        if eos_tokens is None:
            eos_tokens = [self.eos_token_id]
        if min_len is None:
            min_len = getattr(decode_config, 'min_length', 1)
        outputs = self.llm.generate(
            inputs_embeds=llm_input,
            eos_token_id=eos_tokens,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            stopping_criteria=stopping,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=use_cache,
            past_key_values=kv_cache,
            attention_mask=attn_mask,
            return_dict_in_generate=True,
            output_logits=True,
        )

        return outputs

    def generate(self, src_tokens, args=None):
        '''Generate translation hypothesis during validation to check BLEU score.'''
        bsz = len(src_tokens)
        hypos = []
        src_langs = [self.src_lang for _ in range(bsz)]
        tgt_langs = [self.tgt_lang for _ in range(bsz)]
        # Forward audio encoder
        speech_embeds = self.forward_audio_encoder(src_tokens)
        # Forward adaptor
        speech_embeds = self.forward_adaptor(speech_embeds)
        # Prompt
        batch_prompts = [copy.deepcopy(self.prompt[self.model_task]) for _ in range(bsz)]  # Create a batch of prompts for the LLM
        for i_, prompt in enumerate(batch_prompts):
            prompt[0]['content'] = ''.join(prompt[0]['content'])
            prompt[1]['content'] = prompt[1]['content'].replace("<src_lang>", self.lang_codes[src_langs[i_]])
            prompt[1]['content'] = prompt[1]['content'].replace("<tgt_lang>", self.lang_codes[tgt_langs[i_]])
        batch_prompts = self.llm_tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokenized_prompts = self.llm_tokenizer(batch_prompts, return_tensors='pt').input_ids.to(self.device)  # B, T
        
        # LLM input
        wrapped_embeds = self.wrap_prompt(speech_embeds, tokenized_prompts)  # Wrap speech embeds with prompt embeds to get the LLM input
        
        # forward LLM to get translation text
        stop_words_ids = [torch.tensor([self.eos_token_id]).to(self.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llm_generate(wrapped_embeds, self.decode_config, stopping=stopping_criteria, args=args)
        hypos = outputs['sequences']
        return hypos

    def inference(self, src_tokens, prompt, llm_cache, args=None):
        ''' Speech-to-any inference function during evaluation '''
        # Decoding parameters
        top_p = getattr(self.decode_config, 'top_p', 0.7)
        top_k = getattr(self.decode_config, 'top_k', 10)
        temp = getattr(self.decode_config, 'temperature', 1.0)

        # Prepare prompt
        tokenized = self.llm_tokenizer(prompt, return_tensors='pt').to(self.device)  # B, T
        prompt_tokens = tokenized.input_ids
        
        # Forward audio encoder
        speech_embeds = self.forward_audio_encoder(src_tokens)
        # Forward adaptor
        speech_embeds = self.forward_adaptor(speech_embeds)
        # Warp prompt with speech embeds
        wrapped_embeds = self.wrap_prompt(speech_embeds, prompt_tokens)

        # Forward LLM. Yield text and audio tokens in each autoregressive step.
        max_text_len = getattr(self.decode_config, 'max_new_tokens', 128)
        llm_input_embeds = wrapped_embeds  # Initial input to LLM
        for decode_step, _ in enumerate(range(max_text_len)):
            llm_outputs = self.llm(inputs_embeds=llm_input_embeds, 
                                   output_hidden_states=True, 
                                   use_cache=True,
                                   past_key_values=llm_cache,
                                )
            st_logits = llm_outputs.logits  # B, T, V_t
            next_st_token = st_logits[:, -1:, :].argmax(dim=-1)
            if next_st_token.item() == self.eos_token_id or decode_step >= max_text_len-1:
                break
            else:
                yield next_st_token  # Yield the generated token
            # Update LLM input with the new tokens
            llm_input_embeds = self.llm.model.model.embed_tokens(next_st_token)  # B, 1, C
    
    
    def sampling(self, logits, top_p=0.9, top_k=50, temp=1.0):
        ''' Top-p and top-k sampling '''
        # Flatten all but last dimension
        logits_shape = logits.shape
        vocab_size = logits_shape[-1]
        logits = logits.view(-1, vocab_size)  # [N, V]

        # Apply temperature
        logits = logits / temp
        # Top-k sampling
        if top_k > 0:
            # Get the top k logits and their indices
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            # Set all other logits to -inf
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered
        # Top-p sampling (nucleus sampling)
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # Calculate cumulative probabilities
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token (shift right to keep the first token above threshold)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            # Set removed indices to -inf
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            # Scatter back to original positions
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, sorted_indices, sorted_logits)
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    





@register_model_architecture(model_name="llm_sst", arch_name="speech_to_text_base")
def st_base_arch(args):
    args.model_task = getattr(args, 'model_task', None) if getattr(args, 'model_task', None) is not None else 'st'  # Speech-to-text only
    args.whisper_path = getattr(args, 'whisper_path', None) if getattr(args, 'whisper_path', None) is not None else "openai/whisper-large-v3"  # Path to Whisper model
    args.llm_path = getattr(args, 'llm_path', None) if getattr(args, 'llm_path', None) is not None else "Qwen/Qwen3-8B"  # Path to LLM model
    args.freeze_encoder = getattr(args, 'freeze_encoder', True)  # Whether to freeze Whisper encoder
    args.num_projection_layers = getattr(args, 'num_projection_layers', None) if getattr(args, 'num_projection_layers', None) is not None else 2
    args.lora_rank = getattr(args, 'lora_rank', None) if getattr(args, 'lora_rank', None) is not None else 16  # LoRA matrix rank
    args.lora_alpha = getattr(args, 'lora_alpha', None) if getattr(args, 'lora_alpha', None) is not None else 32  # LoRA scaling alpha
    args.lora_dropout = getattr(args, 'lora_dropout', None) if getattr(args, 'lora_dropout', None) is not None else 0.0  # LoRA dropout rate

    args.decode_method = getattr(args, 'decode_method', None) if getattr(args, 'decode_method', None) is not None else 'beam'  # Decoding method during validation
    logger.info("speech_to_text_base architecture has been registered!")

    




