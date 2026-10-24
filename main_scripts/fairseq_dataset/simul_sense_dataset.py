# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from fairseq.data import ConcatDataset, Dictionary, FairseqDataset, ResamplingDataset
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import S2TDataConfig
from transformers import AutoProcessor
from tqdm import tqdm
logger = logging.getLogger(__name__)

def _is_int_or_np_int(n):
    return isinstance(n, int) or (
        isinstance(n, np.generic) and isinstance(n.item(), int)
    )

class language_code_mapper:
    lang_to_code = {
        'en': "English",
        'de': "German",
        'fr': "French",
        'es': "Spanish",
    }
    lang_to_id = {
        'en': 0,
        'de': 1,
        'fr': 2,
        'es': 3,
    }

@dataclass
class SimulSenseDatasetItem(object):
    index: int
    source: torch.Tensor
    source_text: Optional[torch.Tensor] = None
    lang: Optional[str] = None

class SimulSenseDataset(FairseqDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        tgt_texts: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        langs: Optional[List[str]] = None,
        n_frames_per_step=1,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        self.src_texts = src_texts
        assert src_texts is None or len(src_texts) == self.n_samples
        assert len(n_frames) == self.n_samples > 0
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.ids = ids
        if not langs or any(lang != 'en' for lang in langs):
            raise ValueError('This release accepts English detector training data only')
        self.langs = langs
        self.shuffle = cfg.shuffle if is_train_split else False
        self.n_frames_per_step = n_frames_per_step
        self.tgt_lens = [0] * self.n_samples
        self.tokenizer = AutoProcessor.from_pretrained('openai/whisper-large-v3').tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids('<|nospeech|>')
        logger.info(self.__repr__())

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"n_frames_per_step={self.n_frames_per_step}, "
            f"shuffle={self.shuffle}, "
        )
    
    def get_tokenized_text(self, texts, index: Union[int, List[int]], tokenize=True):
        if _is_int_or_np_int(index):
            text = texts[index]
        else:
            text = " ".join([texts[i] for i in index])
        if tokenize:
            text = self.tokenizer(text).input_ids

        return text

    def _get_source_audio(self, index: Union[int, List[int]]) -> torch.Tensor:
        """
        Gives source audio for given index with any relevant transforms
        applied. For ConcatAug, source audios for given indices are
        concatenated in given order.
        Args:
            index (int or List[int]): index—or in the case of ConcatAug,
            indices—to pull the source audio for
        Returns:
            source audios concatenated for given indices with
            relevant transforms appplied
        """

        if _is_int_or_np_int(index):
            source = get_features_or_waveform(
                self.audio_paths[index],
                need_waveform=self.cfg.use_audio_input,
                use_sample_rate=self.cfg.use_sample_rate,
            )
        else:
            source = np.concatenate(
                [
                    get_features_or_waveform(
                        self.audio_paths[i],
                        need_waveform=self.cfg.use_audio_input,
                        use_sample_rate=self.cfg.use_sample_rate,
                    )
                    for i in index
                ]
            )

        if self.cfg.use_audio_input:
            source = torch.from_numpy(source).float()
            if self.cfg.standardize_audio:
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
        else:
            source = torch.from_numpy(source).float()
        return source

    def __getitem__(self, index: int) -> SimulSenseDatasetItem:
        source = self._get_source_audio(index)
        source = source.numpy()
        assert self.langs is not None
        lang = self.langs[index]
        self.tokenizer.set_prefix_tokens(
            language=lang,
            task="transcribe",
            predict_timestamps=False,
        )
        source_text = self.src_texts[index]
        source_text = source_text.replace('<seg>', '<|nospeech|>')
        source_text = self.tokenizer(source_text).input_ids

        return SimulSenseDatasetItem(
            index=index, 
            source=source, 
            source_text=source_text,
            lang=lang,
        )

    def __len__(self):
        return self.n_samples

    def collater(
        self, samples: List[SimulSenseDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        sources = [x.source for x in samples]

        frames = sources
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.shape[0] for x in sources], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        order_list = order.tolist()
        indices = indices.index_select(0, order)
        frames = [frames[i] for i in order_list]
        
        source_text = [x.source_text for x in samples]
        source_text = [source_text[i] for i in order]

        langs = [x.lang for x in samples]
        langs = [langs[i] for i in order]
        lang_ids = [language_code_mapper.lang_to_id[lang] for lang in langs]
        group_ids = self.get_group_ids(source_text)
        source_text, group_ids, decoder_pad_mask = self.pad_src_text_and_group_ids(source_text, group_ids)
        
        net_input = {
            "src_tokens": frames,
            "group_ids": group_ids,
            "src_texts": source_text,
            "decoder_pad_mask": decoder_pad_mask,
            'langs': lang_ids,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order

        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        return self.n_frames[index], self.tgt_lens[index]

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False
    
    def get_group_ids(self, tokenized_src):
        """
        <sense_unit_1> <seg><sense_unit_2> <seg><sense_unit_3> ...

        returns:
            A List contains which sense unit each token belongs to.
            [1, 1, 1, 2, 2, ...]
        """
        group_ids = []
        for src in tokenized_src:
            src_array = np.array(src, dtype=np.int32)
            is_seg = (src_array == self.seg_token_id).astype(np.int32)
            group_id = np.ones(len(src_array), dtype=np.int32)
            group_id[1:] += np.cumsum(is_seg[:-1])
            # group_id[-1] -= 1  # Do not count the eos token as a new unit
            group_ids.append(group_id.tolist())
        return group_ids

    def pad_src_text_and_group_ids(self, source_text, group_ids):
        # Padding tokenized_src and group_ids
        source_text = [torch.tensor(t, dtype=torch.long) for t in source_text]
        max_len = max(t.size(0) for t in source_text)
        pad_id = self.tokenizer.pad_token_id
        padded_src = nn.utils.rnn.pad_sequence(source_text, batch_first=True, padding_value=pad_id)
        # Create decoder padding mask
        lengths = torch.tensor([t.size(0) for t in source_text])
        max_len = padded_src.size(1)
        decoder_pad_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        decoder_pad_mask = decoder_pad_mask.long()  # (B, max_len)

        group_ids = [torch.tensor(u, dtype=torch.long) for u in group_ids]
        group_ids = nn.utils.rnn.pad_sequence(group_ids, batch_first=True, padding_value=0)  # (B, U)
        return padded_src, group_ids, decoder_pad_mask

class SimulSenseDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_SRC_TEXT = "src_text"
    KEY_LANG = "lang"
    DEFAULT_SRC_TEXT = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        n_frames_per_step,
        max_samples: int = -1,
    ) -> SimulSenseDataset:
        audio_root = Path(cfg.audio_root)
        samples = samples[:max_samples] if max_samples > 0 else samples
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        langs = [s.get(cls.KEY_LANG, None) for s in samples]
        dataset_cls = (SimulSenseDataset)

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
            audio_paths=audio_paths,
            n_frames=n_frames,
            src_texts=src_texts,
            ids=ids,
            langs=langs,
            n_frames_per_step=n_frames_per_step,
        )

        return ds

    @classmethod
    def _load_samples_from_tsv(cls, root: str, split: str):
        tsv_path = Path(root) / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
        return samples
    
    @classmethod
    def get_size_ratios(
        cls, datasets: List[SimulSenseDataset], alpha: float = 1.0
    ) -> List[float]:
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""

        id_to_lp, lp_to_sz = {}, defaultdict(int)
        for ds in datasets:
            lang_pairs = {f"{s}->{t}" for s, t in zip(ds.src_langs, ds.tgt_langs)}
            assert len(lang_pairs) == 1
            lang_pair = list(lang_pairs)[0]
            id_to_lp[ds.split] = lang_pair
            lp_to_sz[lang_pair] += sum(ds.n_frames)

        sz_sum = sum(v for v in lp_to_sz.values())
        lp_to_prob = {k: v / sz_sum for k, v in lp_to_sz.items()}
        lp_to_tgt_prob = {k: v**alpha for k, v in lp_to_prob.items()}
        prob_sum = sum(v for v in lp_to_tgt_prob.values())
        lp_to_tgt_prob = {k: v / prob_sum for k, v in lp_to_tgt_prob.items()}
        lp_to_sz_ratio = {
            k: (lp_to_tgt_prob[k] * sz_sum) / v for k, v in lp_to_sz.items()
        }
        size_ratio = [lp_to_sz_ratio[id_to_lp[ds.split]] for ds in datasets]

        p_formatted = {
            k: f"{lp_to_prob[k]:.3f}->{lp_to_tgt_prob[k]:.3f}" for k in lp_to_sz
        }
        logger.info(f"sampling probability balancing: {p_formatted}")
        sr_formatted = {ds.split: f"{r:.3f}" for ds, r in zip(datasets, size_ratio)}
        logger.info(f"balanced sampling size ratio: {sr_formatted}")
        return size_ratio

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        split: str,
        is_train_split: bool,
        n_frames_per_step,
        max_samples: int = -1,
    ) -> SimulSenseDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            n_frames_per_step,
            max_samples=max_samples,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        is_train_split: bool,
        epoch: int,
        seed: int,
        n_frames_per_step: int = 1,
        max_samples: int = -1,
    ) -> SimulSenseDataset:
        datasets = [
            cls._from_tsv(
                root=root,
                cfg=cfg,
                split=split,
                is_train_split=is_train_split,
                n_frames_per_step=n_frames_per_step,
                max_samples=max_samples,
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
