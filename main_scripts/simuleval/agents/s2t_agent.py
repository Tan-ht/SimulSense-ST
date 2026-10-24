import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simuleval.agents import AgentPipeline
from simuleval.utils import entrypoint
from simulsense_agent_simul import SimulSenseAgent

@entrypoint
class S2TWithSimulSensePipeline(AgentPipeline):
    pipeline = [
        SimulSenseAgent,
    ]

    def __init__(self, args) -> None:
        s2t = self.pipeline[0](args)

        module_list = [s2t]
        super().__init__(module_list)

    @classmethod
    def from_args(cls, args):
        return cls(args)
