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
import importlib
import pkgutil
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional


QUALITY_SCORER_REGISTRY = {}


def register_quality_scorer(name):
    """
    Decorator for registering a quality scorer class.

    Args:
        name (str): The unique identifier for the scorer.

    Raises:
        TypeError: If the decorated class is not a subclass of
            :class:`QualityScorer`.

    Example:
        >>> @register_quality_scorer("bleu")
        ... class BLEUScorer(QualityScorer):
        ...     ...
    """
    def register(cls):
        if not issubclass(cls, QualityScorer):
            raise TypeError(f"Cannot register {cls.__name__}: must be a subclass of QualityScorer")
        QUALITY_SCORER_REGISTRY[name] = cls
        return cls

    return register


@dataclass
class QualityScoringSample:
    """
    Data structure representing a single evaluation sample.

    Attributes:
        audio_name (str): The identifier of the audio file.
        hypothesis (str): The system-generated hypothesis text.
        reference (Optional[List[str]]): One or more reference translations, or ``None`` if not
            required.
        source (Optional[List[str]]): The source transcription or text, or ``None`` if not
            required.
    """
    audio_name: str
    hypothesis: str
    reference: Optional[List[str]] = None
    source: Optional[List[str]] = None


class QualityScorer:
    """
    Abstract base class for all quality scorers.

    A quality scorer evaluates system hypotheses against references and/or source sentences
    and returns a numerical score.

    Subclasses must implement the abstract methods defined here and should be registered via
    :func:`register_quality_scorer`.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def score(self, samples: List[QualityScoringSample]) -> float:
        """
        Compute a quality score over a list of samples.

        Args:
           samples (List[QualityScoringSample]): Samples to be evaluated.

        Returns:
           float: The computed quality score.
        """
        ...

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add scorer-specific arguments to the CLI parser.

        Args:
            parser (argparse.ArgumentParser): The parser to extend.
        """
        ...

    @abstractmethod
    def requires_source(self) -> bool:
        """
        Indicate whether this scorer requires the source text.

        Returns:
            bool: True if source sentences are required, False otherwise.
        """
        ...

    @abstractmethod
    def requires_reference(self) -> bool:
        """
        Indicate whether this scorer requires reference translations.

        Returns:
            bool: True if references are required, False otherwise.
        """
        ...


for loader, name, is_pkg in pkgutil.walk_packages(__path__, __name__ + "."):
    importlib.import_module(name)
