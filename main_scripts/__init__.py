"""Register the released speech-to-text models, tasks, and criteria."""
from . import models
from .tasks import simul_speech_translation, simulsense_task
from .criterions import llm_criterions, simulsense_criterion
