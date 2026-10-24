from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics
import torch.nn.functional as F
import torch
import sacrebleu

@register_criterion("s2t_criterion")
class S2TCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        # Use the loss provided by the model
        outputs = model(**sample["net_input"])
        st_loss = outputs.get('st_loss', None)

        nsentences = (sample["nsentences"])
        logging_output = {
            "loss": st_loss.item(),
            "nsentences": nsentences,
            }

        return st_loss, nsentences, logging_output
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", total_loss_sum / nsentences, nsentences, round=3)
        
        # BLEU
        if logging_outputs[0].get("hypos", None) is not None:
            hyps = []
            refs = []
            for log in logging_outputs:
                hyps.extend(log.get("hypos", []))
                refs.extend(log.get("refs", []))
            if len(hyps) > 0:
                assert len(hyps) == len(refs), "hyps and refs must have the same length"
                target_lang = logging_outputs[0].get("lang", None)
                assert target_lang is not None, "target_lang must be provided"
                if target_lang is not None and target_lang == "ja":
                    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="ja-mecab")
                elif target_lang is not None and target_lang == "zh":
                    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
                else:
                    bleu = sacrebleu.corpus_bleu(hyps, [refs])
                bleu_sum = bleu.score
                total_sentences = len(hyps)
                metrics.log_scalar("bleu", bleu_sum, total_sentences, round=3)
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        When loggging outputs are scalars and can be summed, set this to True to use _fast_stat_sync_sum.
        If the logging outputs are complex types, set this to False.
        """
        return False
    

