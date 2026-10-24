import os
import sys
from argparse import ArgumentParser
from typing import Optional

import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simuleval.agents.actions import Action, ReadAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from st_policy_base import STPolicyAgentBase, STPolicyState


class VADPolicyState(STPolicyState):
    def reset(self):
        super().reset()
        self.last_vad_end_sample = 0


@entrypoint
class VADPolicyAgent(STPolicyAgentBase):
    def __init__(self, args):
        super().__init__(args)
        self.vad_threshold = args.vad_threshold
        self.vad_min_silence_ms = args.vad_min_silence_ms
        self.vad_sample_rate = 16000
        self.vad_model = load_silero_vad()

    def build_states(self) -> AgentStates:
        return VADPolicyState()

    @staticmethod
    def add_args(parser: ArgumentParser):
        STPolicyAgentBase.add_args(parser)
        parser.add_argument(
            "--vad-threshold",
            type=float,
            required=True,
            help="Silero VAD speech probability threshold",
        )
        parser.add_argument(
            "--vad-min-silence-ms",
            type=int,
            default=500,
            help="Minimum trailing silence before treating a VAD segment as ended",
        )

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states

        if states.source_finished:
            return self.write_translation(states)
        if states.source_sample_rate == 0 or len(states.source) == 0:
            return ReadAction()

        speech_timestamps = self.get_speech_timestamps(states)
        if len(speech_timestamps) == 0:
            return ReadAction()

        latest_end = max(ts["end"] for ts in speech_timestamps)
        scale_factor = states.source_sample_rate / self.vad_sample_rate
        latest_end_source_sample = int(latest_end * scale_factor)
        min_silence_source_samples = int(
            states.source_sample_rate * self.vad_min_silence_ms / 1000.0
        )
        has_trailing_silence = len(states.source) - latest_end_source_sample >= min_silence_source_samples
        if latest_end_source_sample > states.last_vad_end_sample and has_trailing_silence:
            states.last_vad_end_sample = latest_end_source_sample
            return self.write_translation(states)
        return ReadAction()

    def get_speech_timestamps(self, states: AgentStates):
        audio = torch.tensor(states.source, dtype=torch.float32)
        sample_rate = states.source_sample_rate
        if sample_rate != self.vad_sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.vad_sample_rate)
            sample_rate = self.vad_sample_rate
        return get_speech_timestamps(
            audio,
            self.vad_model,
            threshold=self.vad_threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=self.vad_min_silence_ms,
        )
