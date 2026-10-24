"""Portable configuration helpers; importing this module does not load models."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('zh', 'de', 'ja')


def validate_pair(source, target):
    if source != 'en' or target not in TARGETS:
        raise ValueError('Supported directions: en-zh, en-de, en-ja')


def checkpoint_file_in(path):
    path = Path(path)
    if path.is_file() and path.suffix == '.pt':
        return str(path)
    candidates = sorted(path.glob('*.pt')) if path.is_dir() else []
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one translation .pt checkpoint in {path}; found {len(candidates)}')
    return str(candidates[0])


def relocate_checkpoint(state, source, target):
    """Replace saved machine paths while preserving model architecture settings."""
    validate_pair(source, target)
    from omegaconf import OmegaConf, open_dict
    from contextlib import nullcontext

    def update(config, values):
        with open_dict(config) if OmegaConf.is_config(config) else nullcontext():
            for key, value in values.items():
                if isinstance(config, dict):
                    config[key] = value
                else:
                    setattr(config, key, value)

    def get(config, key):
        return config.get(key) if hasattr(config, 'get') else getattr(config, key, None)

    task, model = state['cfg']['task'], state['cfg']['model']
    if get(task, '_name') not in (None, 'llm_s2t_task'):
        raise ValueError('Checkpoint is not a released ST translation model')
    if (get(model, 'src_lang'), get(model, 'tgt_lang')) != (source, target):
        raise ValueError('Checkpoint language pair does not match the requested direction')
    update(task, {'user_dir': str(ROOT / 'main_scripts'),
                  'config_yaml': 'config/speech_llm.yaml', 'data': str(ROOT / 'data')})
    update(model, {'user_dir': str(ROOT / 'main_scripts'),
                   'prompt_path': str(ROOT / 'main_scripts/prompts/train_prompt.json'),
                   'decode_config': str(ROOT / 'main_scripts/config/decode_config.yaml'),
                   'whisper_path': os.environ.get('SIMULSENSE_WHISPER_PATH', 'openai/whisper-large-v3'),
                   'llm_path': os.environ.get('SIMULSENSE_LLM_PATH', 'Qwen/Qwen3-8B'),
                   'model_task': 'st', 'inference': True})
