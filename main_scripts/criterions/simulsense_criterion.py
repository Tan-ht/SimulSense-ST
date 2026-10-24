import torch.nn.functional as F
import torch
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics
from jiwer import wer
import torch.distributed as dist

@register_criterion("simulsense_criterion")
class SimulSenseCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        outputs = model(**sample["net_input"])
        logits = outputs['logits']
        targets = outputs['targets']
        unit_ids = outputs['unit_ids']
        weights = outputs['weights']
        weights_quantity = outputs['weights_quantity']
        target_quantity = outputs['target_quantity']
        ctc_logits = outputs['ctc_logits']
        enc_pad_mask = outputs['encoder_padding_mask']

        is_master = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if self.training and is_master and model.num_updates % 100 == 0:
            self.log_train_weights(weights, step=model.num_updates)
        
        # Shift logits and targets for cross-entropy loss
        logits = logits[:, :-1, :]
        targets = targets[:, 1:]
        unit_ids = unit_ids[:, 1:]
        # Compute losses
        # unit_level_ce_loss = self.compute_per_unit_loss(logits, targets, unit_ids)
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction='mean')
        quantity_loss = F.smooth_l1_loss(weights_quantity, target_quantity)
        input_lengths = enc_pad_mask.sum(dim=1).long()  # (B,)
        special_tokens = model.tokenizer.all_special_ids
        ctc_targets, target_lengths = self.get_ctc_targets(targets, special_tokens)
        if ctc_logits is None:
            ctc_loss = 0.0
        else:
            ctc_logits = ctc_logits.float()
            ctc_loss = F.ctc_loss(
                F.log_softmax(ctc_logits, dim=-1).transpose(0, 1),  # (T, B, V)
                ctc_targets,
                input_lengths,
                target_lengths,
                blank=model.tokenizer.pad_token_id,
                reduction='mean',
                zero_infinity=True,
            )
            ctc_loss = ctc_loss.item()
        # quantity_loss = 0.0
        lambda_quantity = 1.0
        quantity_loss = lambda_quantity * quantity_loss
        loss = ce_loss + ctc_loss + quantity_loss
        sample_size = sample["nsentences"]
        logging_output = {
            "loss": loss.item(),
            "ctc_loss": ctc_loss,
            "ce_loss": ce_loss.item(),
            "quantity_loss": quantity_loss.item(),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def get_ctc_targets(self, tokens, special_tokens):
        """Extract CTC targets from the sample."""
        if not isinstance(special_tokens, torch.Tensor):
            special_tokens = torch.tensor(list(special_tokens), device=tokens.device, dtype=tokens.dtype)
        else:
            special_tokens = special_tokens.to(tokens.device)
        mask_valid = (tokens != -100)
        if special_tokens.numel() > 0:
            is_special = torch.isin(tokens, special_tokens)
            mask_valid = mask_valid & (~is_special)
        target_lengths = mask_valid.sum(dim=1)  # (B,)
        ctc_targets = torch.masked_select(tokens, mask_valid)

        return ctc_targets, target_lengths.long()

    def compute_per_unit_loss(self, logits, labels, unit_ids):
        B, T_dec, V = logits.shape
        total_loss = 0.0
        total_units = 0

        for b in range(B):
            batch_unit_ids = unit_ids[b]      # (T_dec,)
            batch_labels = labels[b]          # (T_dec,)
            batch_logits = logits[b]          # (T_dec, V)
            
            max_unit = batch_unit_ids.max().item()
            if max_unit <= 0:
                continue  # no valid units in this sample
            
            for u in range(1, max_unit + 1):
                # Create mask for unit u, excluding padding (u>0 already ensured) and ignored labels
                mask = (batch_unit_ids == u) & (batch_labels != -100)  # (T_dec,)
                num_tokens = mask.sum().item()
                if num_tokens == 0:
                    continue
                
                unit_logits = batch_logits[mask]   # (num_tokens, V)
                unit_labels = batch_labels[mask]   # (num_tokens,)
                loss_u = F.cross_entropy(unit_logits, unit_labels, reduction='mean')
                total_loss += loss_u
                total_units += 1

        if total_units == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return total_loss / total_units
    
    def log_train_weights(self, weights, step):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        weights = weights[0].float().detach().cpu().numpy()
        wandb.log(
            {
                "dist/weights": wandb.Histogram(weights),
            },
            step=step,
        )
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        quantity_loss_sum = sum(log.get("quantity_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
            metrics.log_scalar("ce_loss", ce_loss_sum / sample_size, sample_size, round=3)
            metrics.log_scalar("ctc_loss", ctc_loss_sum / sample_size, sample_size, round=3)
            metrics.log_scalar("quantity_loss", quantity_loss_sum / sample_size, sample_size, round=3)
        
            if logging_outputs[0].get("hypo", None) is not None:
                hyps = []
                refs = []
                for log in logging_outputs:
                    hyps.append(log.get("hypo", ''))
                    refs.append(log.get("ref", ''))
                if len(hyps) > 0:
                    assert len(hyps) == len(refs), "hyps and refs must have the same length"
                wer_score = wer(refs, hyps)*100
                total_sentences = len(hyps)
                metrics.log_scalar("wer", wer_score, total_sentences, round=3)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        When loggging outputs are scalars and can be summed, set this to True to use _fast_stat_sync_sum.
        If the logging outputs are complex types, set this to False.
        """
        return False
