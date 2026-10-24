import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairseq.tasks import FairseqTask, register_task
import logging
import torch
from pathlib import Path
from fairseq.data import Dictionary
from fairseq_dataset.speech_to_text_dataset import (
    SpeechToTextDatasetCreator,
    SpeechToTextDataset,
    S2TDataConfig
    )
import os

logger = logging.getLogger(__name__)

@register_task("llm_s2t_task")
class LlmS2T(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename",
        )
        parser.add_argument('--max-source-positions', type=int, default=1e6, help='Maximum input length')
        parser.add_argument('--max-target-positions', type=int, default=1024, help='Maximum output length')

    def __init__(self, args, data_cfg=None, tgt_dict=None):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.args = args
        self.valid_num = 0
        self.data_cfg = data_cfg
        self.speaker_to_id = self._get_speaker_to_id()
        self.model = None
        self.audio_input = True  # Ensure src_tokens on cpu in data loading
        self.simul_finetune = getattr(args, "simul_finetune", False)

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a given dataset split (train/valid/test)."""
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict,
            pre_tokenizer=None,
            bpe_tokenizer=self.model.llm_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
            multitask=None,
        )
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.user_dir) / args.config_yaml)
        dict_path = Path(args.user_dir) / 'llm_vocab' / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix(), add_special_symbols=False)
        tgt_dict.pad_index = tgt_dict.index("<|endoftext|>")
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, data_cfg, tgt_dict)

    @property
    def target_dictionary(self):
        return self.tgt_dict
    
    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.args.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id
    
    def max_positions(self):
        """
        Return the maximum source and target positions.
        """
        return (self.args.max_source_positions, self.args.max_target_positions)
    
    def build_model(self, args):
        """
        Build the model.
        """
        model = super().build_model(args)
        self.model = model
        
        return self.model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        """
        Override to train using audio features and text.
        """
        self.valid_num = 0
        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        # Inference for BLEU and WER during validation
        if self.valid_num < 100:
            hypos, refs = self.model_inference(model, sample)
            if self.valid_num < 5:
                logger.info("example hypothesis: " + hypos[0] + "\n")
                logger.info("example reference: " + refs[0] + "\n")
            logging_output["hypos"] = hypos
            logging_output["refs"] = refs
            logging_output['lang'] = sample['net_input']['tgt_langs'][0]
            self.valid_num += 1

        return loss, sample_size, logging_output

    def model_inference(self, model, sample):
        tgt_refs = sample['net_input']['target_texts']
        src_tokens = sample['net_input']['src_tokens']
        st_hypos = model.generate(src_tokens, args=self.args)
        translations = model.decode_fn(st_hypos)
        return translations, tgt_refs

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    
