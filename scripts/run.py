#!/usr/bin/env python3
"""Training and evaluation entry points for en-zh, en-de, and en-ja."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('zh', 'de', 'ja')


def existing(path):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise argparse.ArgumentTypeError(f'Not found: {p}')
    return p


def positive(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be greater than zero')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('train-st', 'train-detector', 'evaluate', 'stream'):
        q = sub.add_parser(name)
        q.add_argument('--dry-run', action='store_true', help='Print commands without loading models')
        q.add_argument('--output', type=Path, required=True)
        if name != 'train-detector':
            q.add_argument('--target', choices=TARGETS, required=True)
        if name.startswith('train'):
            q.add_argument('--data', type=existing, required=True)
            q.add_argument('--devices', default='0', help='Comma-separated CUDA device indices')
            q.add_argument('--train-subset', default='train')
            q.add_argument('--valid-subset', default='dev')
            q.add_argument('--max-update', type=int, default=100000)
            q.add_argument('--workers', type=int, default=8)
            q.add_argument('--master-port', type=int, default=29500)
        else:
            q.add_argument('--checkpoint', type=existing, required=True)
            q.add_argument('--detector', type=existing)
            q.add_argument('--device', type=int, default=0)
            q.add_argument('--decode', choices=('greedy','beam','sample'), default='greedy')
            q.add_argument('--threshold', type=positive, default=1.0)
            q.add_argument('--chunk-ms', type=positive, default=2000)
            q.add_argument('--source', type=existing, required=True)
            q.add_argument('--reference', type=existing, required=True)
            if name == 'evaluate':
                q.add_argument('--policy', choices=('simulsense','fixed','waitk','vad'), default='simulsense')
                q.add_argument('--waitk', type=int, default=3)
                q.add_argument('--vad-threshold', type=float, default=0.5)
            else:
                q.add_argument('--audio-definition', type=existing, required=True)
                q.add_argument('--transcript', type=existing, help='Optional transcript file for COMET')
    return p


def build_commands(a):
    output = str(a.output.expanduser().resolve())
    if a.command.startswith('train'):
        if not a.devices or any(not d.isdigit() for d in a.devices.split(',')):
            raise ValueError('--devices must contain CUDA indices, for example 0 or 0,1')
        for split in (a.train_subset+','+a.valid_subset).split(','):
            if not (a.data / f'{split}.tsv').is_file():
                raise ValueError(f'Missing dataset split: {split}.tsv')
        if not all(s.startswith('train') for s in a.train_subset.split(',')):
            raise ValueError('Training split names must start with train')
        st = a.command == 'train-st'
        trainer = 's2t_trainer' if st else 'SimulSense_train'
        cmd = [sys.executable,'-m','torch.distributed.run',f'--nproc_per_node={len(a.devices.split(","))}',
               '--master_addr=127.0.0.1',f'--master_port={a.master_port}',str(ROOT / 'main_scripts' / trainer / 'train.py'),
               str(a.data),'--user-dir',str(ROOT/'main_scripts'),
               '--task','llm_s2t_task' if st else 'simulsense_task',
               '--arch','speech_to_text_base' if st else 'mlp',
               '--criterion','s2t_criterion' if st else 'simulsense_criterion',
               '--config-yaml','config/speech_llm.yaml' if st else 'config/sud_config.yaml',
               '--save-dir',output,'--train-subset',a.train_subset,'--valid-subset',a.valid_subset,
               '--optimizer','adam','--adam-betas','(0.9, 0.98)','--lr','1e-4',
               '--lr-scheduler','inverse_sqrt','--warmup-updates','5000','--warmup-init-lr','2e-6',
               '--stop-min-lr','1e-6','--weight-decay','0.05','--max-tokens','400000' if st else '500000',
               '--max-sentences','15' if st else '32','--max-source-positions','200000' if st else '500000',
               '--max-target-positions','1024','--max-update',str(a.max_update),'--max-epoch','30',
               '--patience','10' if st else '20','--validate-interval-updates','2000',
               '--save-interval-updates','2000','--keep-best-checkpoints','10',
               '--best-checkpoint-metric','bleu' if st else 'wer','--seed','0',
               '--num-workers',str(a.workers),'--bf16','--ddp-backend','pytorch_ddp',
               '--skip-invalid-size-inputs-valid-test']
        if st:
            cmd += ['--src-lang','en','--tgt-lang',a.target,'--model-task','st',
                    '--prompt-path',str(ROOT/'main_scripts/prompts/train_prompt.json'),
                    '--decode-config',str(ROOT/'main_scripts/config/decode_config.yaml'),
                    '--maximize-best-checkpoint-metric','--freeze-encoder']
        return [cmd], None
    if not a.checkpoint.is_dir() or len(list(a.checkpoint.glob('*.pt'))) != 1 or not (a.checkpoint/'adapter_model.safetensors').is_file():
        raise ValueError('Translation checkpoint directory needs exactly one .pt and adapter_model.safetensors')
    needs_detector = a.command == 'stream' or a.policy == 'simulsense'
    if needs_detector:
        if a.detector is None or any(not (a.detector / name).is_file() for name in ('sense_units_detector.pt','other_states.pt')):
            raise ValueError('--detector must contain sense_units_detector.pt and other_states.pt')
    unit, tokenizer = ('word','13a') if a.target == 'de' else ('char','zh' if a.target == 'zh' else 'ja-mecab')
    if a.command == 'evaluate':
        agent = {'simulsense':'s2t_agent','fixed':'fixed_policy_agent','waitk':'waitk_policy_agent','vad':'vad_policy_agent'}[a.policy]
        cmd=['simuleval','--agent',str(ROOT/f'main_scripts/simuleval/agents/{agent}.py'),
             '--user-dir',str(ROOT/'main_scripts'),'--source',str(a.source),'--target',str(a.reference),
             '--source-lang','en','--target-lang',a.target,'--model-path',str(a.checkpoint),
             '--device',str(a.device),'--bf16','--computation-aware','--decode-method',a.decode,
             '--no-repeat-ngram-size','3','--source-segment-size',str(a.chunk_ms),
             '--eval-latency-unit',unit,'--sacrebleu-tokenizer',tokenizer,'--output',output]
        if a.policy == 'simulsense': cmd += ['--simul-sense',str(a.detector),'--audio-end-thred',str(a.threshold)]
        if a.policy == 'waitk': cmd += ['--waitk',str(a.waitk)]
        if a.policy == 'vad': cmd += ['--vad-threshold',str(a.vad_threshold)]
        return [cmd], None
    if a.chunk_ms >= 30000:
        raise ValueError('Streaming chunk duration must be less than 30000 ms')
    config={'type':'main_scripts.simulstream.processor.simulsense_processor.SimulSenseProcessor',
            'ckpt_path':str(a.checkpoint),'simulsense_path':str(a.detector),
            'source_lang':'en','target_lang':a.target,'device':a.device,'threshold':a.threshold,
            'speech_chunk_size':a.chunk_ms/1000,'audio_max_duration':30,'max_wait_duration':6.0,
            'detokenizer_type':'simuleval','hf_model_name':os.environ.get('SIMULSENSE_LLM_PATH','Qwen/Qwen3-8B'),
            'latency_unit':unit,'decode_config':{'decode_method':a.decode,
            'greedy':{'max_new_tokens':32,'do_sample':False},
            'beam':{'max_new_tokens':32,'beam_num':4,'do_sample':False,'temperature':1.0},
            'sample':{'max_new_tokens':32,'do_sample':True,'min_length':1,'temperature':0.6,'top_p':0.9,'top_k':10,'seed':0}}}
    config_path=str(Path(output)/'processor.yaml'); log=str(Path(output)/'predictions.jsonl')
    cmds=[['simulstream_inference','--speech-processor-config',config_path,'--wav-list-file',str(a.source),
           '--src-lang','en','--tgt-lang',a.target,'--metrics-log-file',log],
          ['simulstream_score_latency','--scorer','stream_laal','--eval-config',config_path,'--log-file',log,
           '--reference',str(a.reference),'--audio-definition',str(a.audio_definition),'--latency-unit',unit],
          ['simulstream_score_quality','--scorer','sacrebleu','--eval-config',config_path,'--log-file',log,
           '--references',str(a.reference),'--audio-definition',str(a.audio_definition),'--seg-unit',unit,'--tokenizer',tokenizer],
          ['simulstream_stats','--eval-config',config_path,'--log-file',log]]
    if a.transcript:
        cmds.append(['simulstream_score_quality','--scorer','comet','--eval-config',config_path,'--log-file',log,
                     '--references',str(a.reference),'--transcripts',str(a.transcript),
                     '--audio-definition',str(a.audio_definition),'--seg-unit',unit])
    return cmds, config


def main():
    p=parser(); a=p.parse_args()
    try:
        cmds, config=build_commands(a)
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError('Choose a new or empty output directory to avoid mixing results')
    except ValueError as error:
        p.error(str(error))
    for cmd in cmds: print(shlex.join(cmd),flush=True)
    if a.dry_run:
        if config: print(json.dumps(config,indent=2))
        return
    a.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('WANDB_MODE','disabled')
    if a.command.startswith('train'): env['CUDA_VISIBLE_DEVICES']=a.devices
    if config:
        import yaml
        (a.output/'processor.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for index, cmd in enumerate(cmds):
        if a.command == 'stream' and index > 0:
            with (a.output/f'{cmd[0]}.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        else:
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)


if __name__ == '__main__': main()
