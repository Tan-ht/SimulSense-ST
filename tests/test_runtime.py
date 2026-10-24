"""Optional integration checks after installing the release dependencies."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=all(importlib.util.find_spec(name) is not None for name in ('torch','transformers','fairseq','simuleval'))


@unittest.skipUnless(RUNTIME,'Install the release environment for runtime integration checks')
class RuntimeTests(unittest.TestCase):
    def test_evaluation_commands_parse_with_real_toolkit(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        sys.path.insert(0,str(ROOT/'main_scripts/simuleval/agents'))
        import run
        from simuleval import options
        from simulsense_agent_simul import SimulSenseAgent
        from fixed_policy_agent import FixedPolicyAgent
        from waitk_policy_agent import WaitKPolicyAgent
        from vad_policy_agent import VADPolicyAgent
        classes={'simulsense':SimulSenseAgent,'fixed':FixedPolicyAgent,'waitk':WaitKPolicyAgent,'vad':VADPolicyAgent}
        with tempfile.TemporaryDirectory(prefix='st runtime ') as temp:
            root=Path(temp);checkpoint=root/'model';checkpoint.mkdir()
            for name in ('model.pt','adapter_model.safetensors'):(checkpoint/name).touch()
            detector=root/'detector';detector.mkdir()
            for name in ('sense_units_detector.pt','other_states.pt'):(detector/name).touch()
            source=root/'source.txt';source.write_text('/example.wav\n')
            ref=root/'target.txt';ref.write_text('example\n')
            for target in ('zh','de','ja'):
                for policy,cls in classes.items():
                    args=run.parser().parse_args(['evaluate','--target',target,'--policy',policy,
                        '--checkpoint',str(checkpoint),'--detector',str(detector),
                        '--source',str(source),'--reference',str(ref),'--output',str(root/'out')])
                    commands,_=run.build_commands(args);argv=commands[0][1:]
                    parser=options.general_parser()
                    options.add_evaluator_args(parser)
                    options.add_scorer_args(parser,argv)
                    options.add_dataloader_args(parser,argv)
                    cls.add_args(parser)
                    parsed=parser.parse_args(argv)
                    self.assertEqual(parsed.source_segment_size,2000.0)
                    self.assertEqual(parsed.target_lang,target)


if __name__=='__main__':unittest.main()
