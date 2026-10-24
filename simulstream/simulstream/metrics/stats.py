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
import json
import logging
from abc import abstractmethod

import simulstream
from simulstream.config import yaml_config
from simulstream.metrics.readers import LogReader, text_items


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.stats')


class Stats:
    """
    Abstract base class for defining evaluation statistics.

    Subclasses must implement:
    - :meth:`name`: unique identifier of the statistic.
    - :meth:`description`: a human-readable explanation.
    - :meth:`compute`: logic to compute the metric from a :class:`LogReader`.
    """
    @abstractmethod
    def name(self) -> str:
        """The unique name of the statistic."""
        ...

    @abstractmethod
    def description(self) -> str:
        """The human-readable explanation of the statistic."""
        ...

    @abstractmethod
    def compute(self, log_reader: LogReader) -> float:
        """
        Compute the value of the statistic.

        Args:
            log_reader (LogReader): Reader object encapsulating log data.

        Returns:
            float: The computed value of the statistic.
        """
        ...


class NormalizedErasure(Stats):
    """
    Compute the **Normalized Erasure** metric.

    This measures the amount of flickering in retranslation, as defined in
    `Arivazhagan et al., "Re-translation versus Streaming for Simultaneous Translation"
     IWSLT 2020 <https://aclanthology.org/2020.iwslt-1.27/>`_.

    It is defined as the ratio:

    .. math::

        \\text{Normalized Erasure} =
        \\frac{\\text{# Deleted Tokens}}{\\text{# Final Generated Tokens}}
    """

    def name(self) -> str:
        return "normalized_erasure"

    def description(self) -> str:
        return "Normalized erasure, defined in https://aclanthology.org/2020.iwslt-1.27/, " \
               "measures flickering in retranslation. It is defined as the ratio between the " \
               "number of tokens that have been deleted and the number of final generated tokens."

    def compute(self, log_reader: LogReader) -> float:
        total_length = 0
        for _, final_text in log_reader.final_outputs().items():
            total_length += len(text_items(final_text, latency_unit=log_reader.latency_unit))
        return log_reader.num_deleted_tokens() / total_length


class RealTimeFactor(Stats):
    """
    Compute the **Real Time Factor**.

    This measures how many seconds of computation are required on average
    for each second of input audio.

    Values greater than 1 indicate that the system is slower than real time
    and cannot process input before the next audio chunk arrives.
    """
    def name(self) -> str:
        return "real_time_factor"

    def description(self) -> str:
        return "The Real Time Factor measures the average computational cost, ie. time in " \
               "seconds spent in computation, for each input audio second. Values higher than 1 " \
               "mean that the system is not able to process the input in time before the next " \
               "input arrives."

    def compute(self, log_reader: LogReader) -> float:
        total_audio_lengths = sum(
            logs[-1]["total_audio_processed"] for _, logs in log_reader.outputs_by_audio.items())
        total_computational_cost = sum(
            sum(log["computation_time"] for log in logs)
            for _, logs in log_reader.outputs_by_audio.items())
        return total_computational_cost / total_audio_lengths


def main(args: argparse.Namespace):
    """
    Main entry point for computing statistics.

    Loads the evaluation configuration and log file, computes all defined
    statistics, and prints them in JSON format.

    Args:
       args (argparse.Namespace): Parsed command-line arguments.
    """
    LOGGER.info(f"Loading evaluation configuration from {args.eval_config}")
    eval_config = yaml_config(args.eval_config)
    LOGGER.info(f"Loading evaluation log file from {args.log_file}")
    log_reader = LogReader(eval_config, args.log_file, latency_unit=args.latency_unit)

    LOGGER.info("Computing stats")
    stats_classes = [NormalizedErasure(), RealTimeFactor()]
    stats = {
        stat.name(): {"description": stat.description(), "value": stat.compute(log_reader)}
        for stat in stats_classes
    }
    print(f"Stats: {json.dumps(stats, indent=4)}")


def cli_main():
    """
    Module for computing evaluation statistics from Simulstream logs.

    This script provides a CLI interface to compute metrics that describe the behavior of
    streaming systems. Metrics are computed from JSONL log files generated during evaluation and
    include:

    - **Normalized Erasure**: measures flickering in retranslation processors.
    - **Computational Cost**: measures average computation time per second of audio.

    The output is printed on standard output in JSON format.

    Typical usage from the command line:

        $ python -m simulstream.metrics.stats --eval-config config/speech_processor.yaml \\
            --log-file metrics.jsonl
    """
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("stats")
    parser.add_argument(
        "--eval-config", type=str, required=True,
        help="Path to the yaml config file containing information about the tokenizer to be used.")
    parser.add_argument(
        "--log-file", type=str, required=True,
        help="Path to the log file with the metrics to be used for the evaluation.")
    parser.add_argument(
        "--latency-unit", choices=["word", "char"], default="word",
        help="Whether to computed stats based on words or characters. Default: word.")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
