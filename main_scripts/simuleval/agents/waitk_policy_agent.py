import os
import sys
from argparse import ArgumentParser
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simuleval.agents.actions import Action, ReadAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from st_policy_base import STPolicyAgentBase, STPolicyState


class WaitKPolicyState(STPolicyState):
    def reset(self):
        super().reset()
        self.last_written_chunk = 0


@entrypoint
class WaitKPolicyAgent(STPolicyAgentBase):
    def __init__(self, args):
        super().__init__(args)
        self.waitk = args.waitk
        self.source_segment_size = args.source_segment_size

    def build_states(self) -> AgentStates:
        return WaitKPolicyState()

    @staticmethod
    def add_args(parser: ArgumentParser):
        STPolicyAgentBase.add_args(parser)
        parser.add_argument("--waitk", type=int, required=True, help="Number of source chunks to wait")

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states

        if states.source_finished:
            return self.write_translation(states)
        if states.source_sample_rate == 0 or len(states.source) == 0:
            return ReadAction()

        current_ms = len(states.source) * 1000.0 / states.source_sample_rate
        current_chunk = int(current_ms // self.source_segment_size)
        if current_chunk < self.waitk:
            return ReadAction()
        if current_chunk > states.last_written_chunk:
            states.last_written_chunk = current_chunk
            return self.write_translation(states)
        return ReadAction()
