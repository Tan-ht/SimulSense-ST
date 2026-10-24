# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import logging
import sys
from typing import List

from sacrebleu import BLEU

from simulstream.metrics.scorers.quality import register_quality_scorer
from simulstream.metrics.scorers.quality.mwersegmenter import MWERSegmenterBasedQualityScorer, \
    ResegmentedQualityScoringSample

try:
    import sacrebleu
except ImportError:
    sys.exit("Please install comet first with `pip install sacrebleu`.")


LOGGER = logging.getLogger('simulstream.metrics.scorers.latency.stream_laal')


@register_quality_scorer("sacrebleu")
class SacreBLEUScorer(MWERSegmenterBasedQualityScorer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.bleu = BLEU(tokenize=args.tokenizer)

    def _do_score(self, samples: List[ResegmentedQualityScoringSample]) -> float:
        hypotheses = []
        references = []
        for sample in samples:
            hypotheses.extend(sample.hypothesis)
            references.extend(sample.reference)
        score = self.bleu.corpus_score(hypotheses, [references])
        LOGGER.info(f"SacreBLEU signature: {self.bleu.get_signature()}")
        LOGGER.info(f"SacreBLEU detailed score: {score}")
        return score.score

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--tokenizer", choices=sacrebleu.metrics.METRICS['BLEU'].TOKENIZERS,
            default=sacrebleu.metrics.METRICS['BLEU'].TOKENIZER_DEFAULT)

    def requires_source(self) -> bool:
        return False
