import os
import sys
from argparse import ArgumentParser
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simuleval.agents.actions import Action, ReadAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from st_policy_base import STPolicyAgentBase, STPolicyState


class FixedPolicyState(STPolicyState):
    def reset(self):
        super().reset()
        self.last_written_source_len = 0


@entrypoint
class FixedPolicyAgent(STPolicyAgentBase):
    def build_states(self) -> AgentStates:
        return FixedPolicyState()

    @staticmethod
    def add_args(parser: ArgumentParser):
        STPolicyAgentBase.add_args(parser)

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states
        if states.source_finished:
            return self.write_translation(states)
        if states.source_sample_rate == 0 or len(states.source) == 0:
            return ReadAction()

        if len(states.source) > states.last_written_source_len:
            states.last_written_source_len = len(states.source)
            return self.write_translation(states)
        return ReadAction()
