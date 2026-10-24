"""CPU-only checks for release scope and command construction."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

runner=load('release_runner',ROOT/'scripts/run.py')
helpers=load('release_helpers',ROOT/'main_scripts/release_utils.py')


class ReleaseTests(unittest.TestCase):
    def test_language_scope(self):
        for target in ('zh','de','ja'): helpers.validate_pair('en',target)
        for source,target in [('de','en'),('en','fr'),('en','en')]:
            with self.assertRaises(ValueError):helpers.validate_pair(source,target)

    def test_checkpoint_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            with self.assertRaises(ValueError):helpers.checkpoint_file_in(p)
            (p/'model.pt').touch()
            self.assertEqual(helpers.checkpoint_file_in(p),str(p/'model.pt'))
            (p/'second.pt').touch()
            with self.assertRaises(ValueError):helpers.checkpoint_file_in(p)

    def test_no_project_rl_or_synthesis_registration(self):
        forbidden={'llm_s2s_task','mtp_train','s2s_criterion','mtp_criterion','nar_mtp','simulst_post_rl'}
        for path in (ROOT/'main_scripts').rglob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id.startswith('register_'):
                    for arg in node.args:
                        if isinstance(arg,ast.Constant):self.assertNotIn(arg.value,forbidden,str(path))
        self.assertFalse(any((ROOT/'main_scripts').rglob('*rl*.py')))

    def test_evaluation_commands_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix='release with spaces ') as temp:
            p=Path(temp); ckpt=p/'checkpoint';ckpt.mkdir()
            for name in ('model.pt','adapter_model.safetensors'):(ckpt/name).touch()
            detector=p/'SimulSense';detector.mkdir()
            for name in ('sense_units_detector.pt','other_states.pt'):(detector/name).touch()
            source=p/'source.txt';source.write_text('/example.wav\n')
            target=p/'target.txt';target.write_text('Example\n')
            definition=p/'segments.yaml';definition.write_text('[]\n')
            for lang,tokenizer,unit in [('zh','zh','char'),('de','13a','word'),('ja','ja-mecab','char')]:
                for policy in ('simulsense','fixed','waitk','vad'):
                    args=runner.parser().parse_args(['evaluate','--target',lang,'--checkpoint',str(ckpt),
                         '--detector',str(detector),'--source',str(source),'--reference',str(target),
                         '--output',str(p/'output'),'--policy',policy,'--dry-run'])
                    commands,_=runner.build_commands(args); cmd=commands[0]
                    self.assertEqual(cmd[cmd.index('--sacrebleu-tokenizer')+1],tokenizer)
                    self.assertEqual(cmd[cmd.index('--eval-latency-unit')+1],unit)
                    self.assertEqual(cmd[cmd.index('--source')+1],str(source))
                args=runner.parser().parse_args(['stream','--target',lang,'--checkpoint',str(ckpt),
                    '--detector',str(detector),'--source',str(source),'--reference',str(target),
                    '--audio-definition',str(definition),'--output',str(p/'stream'),'--dry-run'])
                commands,config=runner.build_commands(args)
                self.assertEqual(config['source_lang'],'en');self.assertEqual(config['target_lang'],lang)
                self.assertIn('greedy',config['decode_config']);self.assertEqual(len(commands),4)

    def test_training_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            for name in ('train.tsv','dev.tsv'):(p/name).touch()
            for mode in ('train-st','train-detector'):
                argv=[mode,'--data',str(p),'--output',str(p/'out'),'--devices','0,1','--dry-run']
                if mode=='train-st':argv+=['--target','ja']
                commands,_=runner.build_commands(runner.parser().parse_args(argv))
                self.assertIn('--nproc_per_node=2',commands[0])
                self.assertNotIn('--wandb-project',commands[0])
                if mode=='train-detector': self.assertIn('mlp',commands[0])


if __name__=='__main__':unittest.main()
