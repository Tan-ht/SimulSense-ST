import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def preferred_attn_implementation():
    override = __import__("os").environ.get("SIMULSENSE_ATTN_IMPLEMENTATION")
    if override:
        return override
    try:
        __import__("flash_attn")
    except Exception:
        return "sdpa"
    return "flash_attention_2"

