#!/usr/bin/env python3
"""Download the published SimulSense detector and ST translation weights."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('zh','de','ja','all'),default='all')
    p.add_argument('--output',type=Path,default=Path('checkpoints'))
    p.add_argument('--revision',default='main',help='Use a commit hash for reproducible downloads')
    a=p.parse_args()
    from huggingface_hub import snapshot_download
    targets=('zh','de','ja') if a.target == 'all' else (a.target,)
    snapshot_download('tan-ht/SimulSense',revision=a.revision,local_dir=a.output,
                      allow_patterns=['SimulSense/*']+[f'translation/en-{lang}/*' for lang in targets])


if __name__ == '__main__': main()
