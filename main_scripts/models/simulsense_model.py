import logging
import torch.nn as nn
import torch
from transformers import AutoProcessor
from fairseq.models import register_model, register_model_architecture, BaseFairseqModel
from .modeling_whisper import WhisperForConditionalGeneration
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from .modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from .utils import preferred_attn_implementation
from peft import get_peft_model, LoraConfig
import math

logger = logging.getLogger(__name__)

class language_code_mapper:
    lang_to_code = {
        'en': "English",
        'de': "German",
        'fr': "French",
        'es': "Spanish",
    }
    lang_to_id = {
        'en': 0,
        'de': 1,
        'fr': 2,
        'es': 3,
    }
    id_to_lang = {v: k for k, v in lang_to_id.items()}

class WeightPredictor(nn.Module):
    def __init__(self, encoder_dim, vocab_size=None, num_langs=4, args=None):
        super().__init__()
        self.type = getattr(args, 'weight_predictor_type', 'transformer')
        num_layers = getattr(args, 'weight_predictor_layers', 1)
        hidden_dim = getattr(args, 'weight_predictor_hidden_dim', 4096)
        # self.linear_proj = nn.Linear(encoder_dim, hidden_dim)
        self.linear_proj = nn.Parameter(torch.randn(num_langs, encoder_dim, hidden_dim))
        if self.type == 'transformer':
            config = Qwen3Config()
            assert hidden_dim == config.hidden_size, "Hidden dim must match Qwen3 hidden size."
            config.intermediate_size = getattr(args, 'weight_predictor_ffn_dim', 11008)
            config._attn_implementation = preferred_attn_implementation()
            # Main components
            self.layers = nn.ModuleList(
                [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(num_layers)]
            )
            self.rotary_emb = Qwen3RotaryEmbedding(config=config)
            self.output_norm = nn.LayerNorm(hidden_dim)
        elif self.type == 'mlp':
            self.mlp_layers = nn.ModuleList()
            for _ in range(num_layers):
                mlp_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                self.mlp_layers.append(mlp_layer)
        if vocab_size is not None:
            self.ctc_head = nn.Linear(hidden_dim, vocab_size)

        self.weight_proj = nn.Linear(hidden_dim, 1)
    
    def forward_transformer(self, hidden_states, padding_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            layer_outputs = layer(hidden_states, 
                                  attention_mask=padding_mask,
                                  position_ids=position_ids, 
                                  position_embeddings=position_embeddings)
            hidden_states = layer_outputs[0]
        hidden_states = self.output_norm(hidden_states)
        return hidden_states
    
    def forward_mlp(self, hidden_states):
        for mlp_layer in self.mlp_layers:
            hidden_states = mlp_layer(hidden_states)
        return hidden_states
        
    def forward(self, encoder_states, lang_ids, padding_mask=None):
        expert_linear_proj = self.linear_proj[lang_ids]
        hidden_states = torch.einsum('btc,bcd->btd', encoder_states, expert_linear_proj)  # (B, T, D)
        if self.type == 'transformer':
            hidden_states = self.forward_transformer(encoder_states, padding_mask)
        elif self.type == 'mlp':
            hidden_states = self.forward_mlp(hidden_states)

        if hasattr(self, 'ctc_head'):
            ctc_logits = self.ctc_head(hidden_states)  # (B, T, V)
        else:
            ctc_logits = None
        logits = self.weight_proj(hidden_states).squeeze(-1)  # (B, T)
        if padding_mask is not None:
            logits = logits.masked_fill(~padding_mask.bool(), -1e4)
        weights = torch.sigmoid(logits)
        quantity = weights.sum(dim=-1) # To calculate the quantity loss
        output = {
            "weights": weights,
            "quantity": quantity,
            "ctc_logits": ctc_logits,
        }

        return output

class BoundaryDetector(nn.Module):
    def __init__(self, integrate_thred=1.0, args=None):
        super().__init__()
        self.args = args
        self.integrate_thred = integrate_thred
    
    def aggregate_weights(self, weights, threshold, unit_nums=None):
        """
        Aggregate weights into segments using CIF style.
        
        Args:
            weights: (B, T) boundary detection weights
            unit_nums: (B,) expected number of units for verification
        """
        B, max_len = weights.shape
        device = weights.device
        # Init 
        accumulated = torch.zeros(B, device=device)
        seg_counter = torch.zeros(B, dtype=torch.long, device=device)
        segment_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        spikes = torch.zeros(B, max_len, device=device, dtype=torch.long, requires_grad=False).detach() # Record the segment points

        for t in range(max_len):
            segment_ids[:, t] = seg_counter # Record current segment ID
            # Accumulate weights
            w_t = weights[:, t]
            total = accumulated + w_t
            fire_mask = total >= (threshold - 1e-4) # Fire decision
            
            # Update current accumulated value
            # If fire, subtract threshold; else keep accumulated value
            accumulated = torch.where(fire_mask, total - threshold, total)
            accumulated = torch.clamp(accumulated, min=0.0)

            spikes[:, t] = fire_mask.long().detach()

            # 更新计数器 (只有在有效长度内的 Fire 才有效)
            is_valid = t < max_len
            effective_fire = fire_mask & is_valid
            seg_counter += effective_fire.long()

        # 3. 根据 Segment IDs 聚合数据 (Gather & Pad)
        #    将 (B, T) 的数据重组为 (B, S, T')
        batch_results = []
        max_unit_num = 0 # 记录最大段数 S
        max_unit_len = 0 # 记录段内最大长度 T'
        for b in range(B):
            sample_res = []
            # 获取该样本有效范围内的 IDs
            curr_ids = segment_ids[b, :]
            # 计算该样本有多少段 (ID从0开始，所以总数是 max_id + 1)
            num_segs = curr_ids.max().item() + 1
            max_unit_num = max(max_unit_num, num_segs)
            
            for s in range(num_segs):
                # 找到属于段 s 的所有索引
                mask = (curr_ids == s)
                # 取出对应的 weights
                seg_data = weights[b, :][mask]
                max_unit_len = max(max_unit_len, seg_data.size(0))
                sample_res.append(seg_data)
            
            batch_results.append(sample_res)

        # 4. 转换为 Tensor 并 Padding
        max_unit_len = max(max_unit_len, 1) # 防止全空
        max_unit_num = max(max_unit_num, 1)

        # 输出容器 (B, S, T')
        out_weights = torch.zeros(B, max_unit_num, max_unit_len, device=device, dtype=weights.dtype)
        out_mask = torch.zeros(B, max_unit_num, max_unit_len, device=device, dtype=torch.bool) # Valid=True

        for b in range(B):
            for s, seg_data in enumerate(batch_results[b]):
                l = seg_data.size(0)
                out_weights[b, s, :l] = seg_data
                out_mask[b, s, :l] = True
        
        boundary_positions = self._spikes_to_boundaries(spikes.detach(), max_len)
        num_units = torch.tensor([len(r) for r in batch_results], device=device).long()
        if unit_nums is not None:
            assert num_units.sum().item() == unit_nums.sum().item(), "Mismatch in expected number of units."

        return {
            "unit_weights": out_weights, # (B, S, T')
            "padding_mask": out_mask.long(),         # (B, S, T')
            "num_units": torch.tensor([len(r) for r in batch_results], device=device).long(),  # (B,)
            "spikes": spikes,
            "boundary_positions": boundary_positions,
        }
    
    def _spikes_to_boundaries(self, spikes, max_length):
        """Convert spike train to boundary positions"""
        spikes = spikes.clone()
        B = spikes.size(0)
        nonzero_indices = torch.nonzero(spikes > 0.5, as_tuple=False)
        boundary_positions = [[] for _ in range(B)]
        
        for batch_idx, time_idx in nonzero_indices:
            boundary_positions[batch_idx.item()].append(time_idx.item() + 1)
        
        for idx in range(len(boundary_positions)):
            boundary_positions[idx].insert(0, 0)
            boundary_positions[idx].append(max_length)
        
        return boundary_positions
    
    def create_cross_attn_mask(self, weights, threshold, unit_nums, unit_ids, padding_idx=0):
        outputs = self.aggregate_weights(weights, threshold, unit_nums)
        boundary_positions = outputs['boundary_positions']
        left_context = getattr(self.args, 'left_context')
        
        B, T_enc = weights.shape
        # 获取这一批次最大的 CIF 段数，用于定义矩阵形状
        max_cif_units = max(len(bp) - 1 for bp in boundary_positions)
        
        # 初始化 Alignment Matrix (B, U_cif, T_enc)
        # 默认值为 0
        alignment_matrix = torch.zeros(B, max_cif_units, T_enc, device=weights.device, dtype=weights.dtype)
        
        for b in range(B):
            bp = boundary_positions[b]
            num_segments = len(bp) - 1
            for u in range(num_segments):
                orig_start = bp[u]
                orig_end = bp[u+1]
                expand_start = max(0, orig_start - left_context)
                expand_end = min(T_enc, orig_end)
                valid_len = expand_end - expand_start
                if valid_len > 0:
                    seg_w = weights[b, expand_start:expand_end]
                    alignment_matrix[b, u, expand_start:expand_end] = seg_w

        target_indices = unit_ids - 1
        is_padding = (unit_ids == padding_idx)
        
        # 安全处理：防止 index out of bounds
        safe_indices = target_indices.clamp(min=0, max=max_cif_units - 1)
        
        # 查表 (Gather): 根据 ID 复用上面的 alignment 行
        batch_indices = torch.arange(B, device=weights.device).unsqueeze(1)
        final_alignment = alignment_matrix[batch_indices, safe_indices] # (B, L_dec, T_enc)
        
        # 清除 Padding Token 的 Mask
        final_alignment[is_padding] = 0.0
        eps = 1e-4
        soft_mask = torch.log(final_alignment + eps)
        
        # 屏蔽 0 值区域 (背景/无关联区域)
        zero_mask = (final_alignment == 0)
        soft_mask.masked_fill_(zero_mask, -1e4)

        soft_mask = soft_mask.unsqueeze(1)  # (B, 1, L_dec, T_enc)

        return soft_mask, outputs

@register_model("SimulSense")
class SimulSense(BaseFairseqModel):
    def __init__(self, modules, args):
        super().__init__()
        self.args = args
        self.audio_encoder = modules["audio_encoder"]
        self.audio_decoder = modules["audio_decoder"]
        self.proj_out = modules["proj_out"]
        if getattr(args, 'freeze_encoder'):
            self.freeze_audio_encoder()
            logger.info('Audio Encoder Freezed')

        if getattr(args, 'freeze_decoder'):
            self.freeze_audio_decoder()
            logger.info('Audio Decoder Freezed')
        else:
            if getattr(args, 'inference', False):
                logger.info("Building PEFT model with inference mode.")
                inference=True
            else:
                inference=False
            lora_rank = args.lora_rank
            lora_alpha = args.lora_alpha
            lora_dropout = args.lora_dropout
            lora_config = LoraConfig(
                inference_mode=inference,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj"],
            )
            self.audio_decoder = get_peft_model(self.audio_decoder, lora_config)
            self.audio_decoder.print_trainable_parameters()

        if getattr(args, 'freeze_proj_out', True):
            for param in self.proj_out.parameters():
                param.requires_grad = False
            logger.info('Output Projection Freezed')
        self.feature_extractor = modules["feature_extractor"]
        self.tokenizer = modules["tokenizer"]
        self.weight_predictor = modules["weight_predictor"]
        # Initialize snnTorch-based boundary detector
        integrate_thred = getattr(args, 'integrate_thred')
        self.boundary_detector = BoundaryDetector(integrate_thred, args)
        self.special_tokens = self.tokenizer.all_special_tokens
        self.special_ids = self.tokenizer.all_special_ids
        self.eos_token_id = self.tokenizer.eos_token_id

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    @classmethod
    def build_model(cls, args, task=None):
        # audio_encoder = AudioEncoder(args)
        # decoder = AudioDecoder(args)
        processor = AutoProcessor.from_pretrained(args.whisper_path)
        whisper = WhisperForConditionalGeneration.from_pretrained(
            args.whisper_path, 
            low_cpu_mem_usage=True,
        )
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        
        audio_encoder = whisper.get_encoder()
        audio_decoder = whisper.get_decoder()
        lm_proj = whisper.proj_out
        encoder_dim = audio_encoder.config.d_model
        if getattr(args, 'ctc', True):
            whisper_vocab_size = len(tokenizer)
        else:
            whisper_vocab_size = None

        weight_predictor = WeightPredictor(encoder_dim, whisper_vocab_size, args=args)
        
        modules = {
            "audio_encoder": audio_encoder,
            "audio_decoder": audio_decoder,
            "proj_out": lm_proj,
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer,
            "weight_predictor": weight_predictor,
        }

        return cls(modules, args)

    def freeze_audio_encoder(self):
        for _, param in self.audio_encoder.named_parameters():
            param.requires_grad = False

    def freeze_audio_decoder(self):
        for _, param in self.audio_decoder.named_parameters():
            param.requires_grad = False

    def extract_features(self, waveform, sample_rate=16000):
        processor_out = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_attention_mask=True,
            do_normalize=True,
            device='cpu',
            return_tensors='pt',
        )
        features = processor_out['input_features']  # numpy
        pad_mask = processor_out['attention_mask']
        features = features.to(self.device).to(self.dtype)
        pad_mask = pad_mask.to(self.device).to(self.dtype)
        
        return features, pad_mask
    
    def forward_audio_encoder(self, waveform):
        features, pad_mask = self.extract_features(waveform)  
        # Compute the maximum length after downsampling
        B, T_mel = pad_mask.shape
        T_enc = T_mel // 2  # Whisper Encoder downsamples the input by 2
        downsampled_mask = pad_mask.view(B, T_enc, 2).max(dim=-1)[0]  # (B, 1500)  # Downsample the mask to match the Whisper encoder output
        actual_len = downsampled_mask.sum(dim=-1).max().long().item()  # (B,)  # Actual length of the features without padding

        encoder_states = self.audio_encoder(features).last_hidden_state
        # sliced_states = encoder_states[:, :actual_len, :]
        # sliced_mask = downsampled_mask[:, :actual_len]

        return encoder_states, downsampled_mask

    def normalize_weights(self, weights, target_fire_nums, integrate_thred):
        target_quantity = target_fire_nums * integrate_thred
        weights_sum = weights.sum(dim=-1)  # (B,)
        normalized_scaler = torch.unsqueeze(target_quantity / (weights_sum + 1e-7), dim=-1) # (B, 1)
        weights = weights * normalized_scaler # (B, T)
        
        return weights
    
    @property
    def dtype(self):
        return list(self.parameters())[0].dtype
    
    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total parameters: {total_params}, Trainable parameters: {trainable_params}"

    def forward(self, src_tokens, group_ids, src_texts, decoder_pad_mask, langs, **kwargs):
        # Forward audio encoder
        encoder_states, encoder_padding_mask = self.forward_audio_encoder(src_tokens)  # (B, T, C)

        # Forward weight predictor
        output = self.weight_predictor(encoder_states, langs, padding_mask=encoder_padding_mask)  # (B, T) for sense unit boundary detection; Actual quantity of sense units calculated from the weights; For loss calculation
        weights = output["weights"]  
        quantity = output["quantity"] # Actual weights quantity
        ctc_logits = output["ctc_logits"]
        unit_nums = group_ids.max(dim=-1).values  # (B,) number of sense units for each utterance

        # Apply scaling strategy to the detection weights to
        # make the boundary trigger nums equal to the target_unit_nums-1
        target_fire_nums = (unit_nums - 1).float()
        target_fire_nums += 0.1  # Small bias to avoid under-triggering
        integrate_thred = getattr(self.args, 'integrate_thred')
        # weights = weights.float() # For numerical stability
        normed_weights = self.normalize_weights(weights, target_fire_nums, integrate_thred)

        cross_attn_mask, outputs = self.boundary_detector.create_cross_attn_mask(
            normed_weights, integrate_thred, unit_nums, group_ids
        )
        cross_attn_mask = cross_attn_mask.to(self.dtype)

        decoder_out = self.audio_decoder(
            input_ids=src_texts,
            encoder_hidden_states=encoder_states,
            attention_mask=decoder_pad_mask,
            encoder_attention_mask=cross_attn_mask,
            output_attentions=True, # True to use manual attention, otherwise default sdpa
        )
        attn = decoder_out.cross_attentions
        logits = decoder_out.last_hidden_state
        logits = self.proj_out(logits)
        labels = torch.where(decoder_pad_mask == 1, src_texts, -100)

        outputs = {
            "logits": logits,
            "targets": labels,
            "unit_ids": group_ids,
            "weights": weights,
            "weights_quantity": quantity,
            "target_quantity": target_fire_nums * integrate_thred,
            "ctc_logits": ctc_logits,
            "encoder_padding_mask": encoder_padding_mask,
        }
        return outputs

    def decode_fn(self, token_ids, skip=True):
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        decoded = decoded.replace("<|nospeech|>", " ")
        if skip:
            for token in self.special_tokens:
                decoded = decoded.replace(token, "")
        return decoded.strip()

    def inference(self, src_tokens, lang, **kwargs):
        # Forward audio encoder
        encoder_states, encoder_padding_mask = self.forward_audio_encoder(src_tokens)  # (B, T, C)
        B, T_enc, C = encoder_states.shape
        # Forward weight predictor
        output = self.weight_predictor(encoder_states, [lang])
        weights = output["weights"] # (B, T)
        threshold = getattr(self.args, 'integrate_thred', 1.0)
        outputs = self.boundary_detector.aggregate_weights(weights, threshold)
        bp = outputs['boundary_positions'][0] # List[int]
        # Decoder generation
        self.tokenizer.set_prefix_tokens(
            language=language_code_mapper.id_to_lang[lang],
            task="transcribe",
            predict_timestamps=False,
        )
        input_ids = self.tokenizer("", return_tensors="pt").input_ids.to(self.device)[:, :-1]
        MAX_TOTAL_LEN = 256  # 总长度限制
        MAX_SEGMENT_LEN = 32 # 每个切片最大生成长度
        left_context = getattr(self.args, 'left_context')
        for i in range(len(bp) - 1):
            if input_ids.shape[1] >= MAX_TOTAL_LEN:
                break
            orig_start = bp[i]
            orig_end = bp[i+1]
            # chunk_start = max(0, orig_start-left_context)
            chunk_end = min(T_enc, orig_end)
            state_seg = encoder_states[:, :chunk_end, :]  # (1, T_seg, C)
            for _ in range(MAX_SEGMENT_LEN):
                if input_ids.shape[1] >= MAX_TOTAL_LEN:
                    break
                outputs = self.audio_decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=state_seg,
                    attention_mask=encoder_padding_mask ,
                    output_attentions=True,
                )
                logits = outputs.last_hidden_state
                logits = self.proj_out(logits)
                next_token_logits = logits[:, -1, :] # 取最后一个时间步
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # (B, 1)
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                token_val = next_token_id.item()
                if token_val in self.special_ids:
                    break
        hypo_ids = input_ids[0].tolist()
        inference_output = {
            "hypo_ids": hypo_ids,
            "boundary_positions": bp,
        }
        return inference_output
            
    def visual_attn_mask(self, mask, title="Cross-Attention Mask", save_path=None):
        """
        Visualize a cross-attention mask where:
        - Valid positions have value 0  → shown in light grey
        - Invalid positions have large negative values → shown in gray
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import copy
        # Ensure float32 for plotting
        mask = mask.detach().float().cpu()
        hard_mask_threshold = -1e4 
        is_hard_masked = mask < hard_mask_threshold
        weights = torch.exp(mask)
        weights_np = weights.numpy()
        mask_np = is_hard_masked.numpy()

        weights_masked = np.ma.masked_where(mask_np, weights_np)
        plt.figure(figsize=(12, 6))
        cmap = copy.copy(plt.cm.viridis) # 'viridis', 'plasma', 'Blues' are good
        cmap.set_bad(color='lightgray') # Color for the Hard Mask
        im = plt.imshow(weights_masked, aspect='auto', interpolation='nearest', 
                        cmap=cmap, origin='upper', vmin=0.0) 

        # Add Colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Attention Weight (Linear Scale)', rotation=270, labelpad=15)

        # Labels
        plt.xlabel('Audio Frames', fontsize=12)
        plt.ylabel('Decoder Steps', fontsize=12)
        plt.title(title, fontsize=14, weight='bold', pad=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def visual_cross_attn(self, cross_attn, title="Cross-Attention", save_path=None, batch_idx=0):
        import matplotlib.pyplot as plt
        if not isinstance(cross_attn, torch.Tensor):
            stacked = torch.stack(cross_attn, dim=0).float()  # (Layers, Batch, Heads, text_len, time_step)
            avg_attn = stacked.mean(dim=[0, 2])  # (Batch, text_len, time_step)
        else:
            avg_attn = cross_attn.float()
        attn_map = avg_attn[batch_idx].cpu().detach().numpy()  # (text_len, time_step)
        plt.figure(figsize=(12, 8))
        plt.imshow(attn_map, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Time Step')
        plt.ylabel('Text Position')
        plt.title(title)
        plt.tight_layout()
        save_path = save_path if save_path is not None else f"{title.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        return attn_map
    
@register_model_architecture(model_name="SimulSense", arch_name="no_ctc")
def no_ctc(args):
    args.ctc = getattr(args, 'ctc', False) # Whether to use CTC loss in weight predictor
    mlp(args)

@register_model_architecture(model_name="SimulSense", arch_name="mlp")
def mlp(args):
    args.left_context = getattr(args, 'left_context', 5)
    args.ctc = getattr(args, 'ctc', True) # Whether to use CTC loss in weight predictor
    args.weight_predictor_type = 'mlp'
    args.weight_predictor_layers = getattr(args, 'weight_predictor_layers', 2)
    base_architecture(args)

@register_model_architecture(model_name="SimulSense", arch_name="base_arch")
def base_architecture(args):
    args.lang = getattr(args, 'lang', None)
    # Encoder Decoder (Whisper)
    args.whisper_path = getattr(args, 'whisper_path', "openai/whisper-large-v3")
    args.freeze_encoder = getattr(args, 'freeze_encoder', True)
    args.freeze_decoder = getattr(args, 'freeze_decoder', False)
    args.freeze_proj_out = getattr(args, 'freeze_proj_out', True)
    # Weight predictor
    args.weight_predictor_type = getattr(args, 'weight_predictor_type', 'transformer')
    args.weight_predictor_layers = getattr(args, 'weight_predictor_layers', 1)
    args.hidden_dim = getattr(args, 'weight_predictor_hidden_dim', 4096)
    args.integrate_thred = getattr(args, 'integrate_thred', 1.0) # 1.0 in original CIF. Set larger value as the target fire nums is very samll.
    args.left_context = getattr(args, 'left_context', 0)  # How many frames to look back when creating cross-attention mask
    # LoRA settings
    args.lora_rank = getattr(args, 'lora_rank', 8)  # LoRA matrix rank
    args.lora_alpha = getattr(args, 'lora_alpha', 16)  # LoRA scaling alpha
    args.lora_dropout = getattr(args, 'lora_dropout', 0.0)  # LoRA dropout rate
