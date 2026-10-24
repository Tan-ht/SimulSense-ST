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

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Any, Dict

import yaml

from simulstream.metrics.detokenizers import get_detokenizer


def text_items(text: str, latency_unit: str) -> List[str]:
    """
    Split a text string into items depending on the latency unit.

    Args:
        text (str): The input text string.
        latency_unit (str): The unit for latency measurement. Must be either:
            - ``"word"`` → split on whitespace.
            - ``"char"`` → split into individual characters.

    Returns:
        List[str]: The list of word or character tokens.

    Raises:
        ValueError: If `latency_unit` is not ``"word"`` or ``"char"``.
    """
    if latency_unit == "word":
        words = text.split(" ")
        return [w for w in words if w != '']
    elif latency_unit == "char":
        return list(text)
    else:
        raise ValueError(
            f"Latency unit `{latency_unit}` not supported. Allowed values are `word` and `char`.")


@dataclass
class OutputWithDelays:
    """
    Representation of a final output sequence and its delays.

    Attributes:
        final_text (str): The detokenized output text.
        ideal_delays (List[float]): Latency values relative to processed audio.
        computational_aware_delays (List[float]): Latency values including computation time.
    """
    final_text: str
    ideal_delays: List[float]
    computational_aware_delays: List[float]

    def text_len(self, latency_unit: str) -> int:
        """
        Return the length of the text in the given latency unit.

        Args:
            latency_unit (str): Either ``"word"`` or ``"char"``.

        Returns:
            int: Number of items in the text.
        """
        return len(self.text_items(latency_unit))

    def text_items(self, latency_unit: str) -> List[str]:
        """
        Return the text split into items (words or characters).

        Args:
            latency_unit (str): Either ``"word"`` or ``"char"``.

        Returns:
            List[str]: Tokens in the specified unit.
        """
        return text_items(self.final_text, latency_unit)

    def last_word(self) -> str:
        """
        Return the last word of the text.

        Returns:
            str: The last word token.
        """
        return self.text_items("word")[-1]


@dataclass
class ReferenceSentenceDefinition:
    """
    Stores the information about a reference sentence.

    Attributes:
        content (str): The sentence text.
        start_time (float): Start time (in seconds) of the segment.
        duration (float): Duration (in seconds) of the segment.
    """
    content: str
    start_time: float
    duration: float


class LogReader:
    """
    Reads and processes JSONL metric logs written by the websocket server.

    This class rebuilds the final outputs (ignoring retranslated tokens) and provides access to
    fine-grained information.

    Args:
        config (SimpleNamespace): Configuration namespace, used for detokenizer setup.
        filepath (str): Path to the log file (JSONL format).
        latency_unit (str, optional): Latency measurement unit, ``"word"`` or ``"char"``.
    """
    def __init__(self, config: SimpleNamespace, filepath: str, latency_unit: str = "word"):
        self.filepath = filepath
        self.detokenizer = get_detokenizer(config)
        self.outputs_by_audio = self._get_outputs()
        self.latency_unit = latency_unit

    def _get_outputs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group outputs by audio file.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping of audio name → list of log entries.
        """
        outputs_by_audio = OrderedDict()
        audio_id_map = {}
        for line in self._read_all():
            if 'metadata' in line:
                audio_id_map[line['id']] = Path(line['metadata']['wav_name']).stem
            elif 'id' in line:
                assert line['id'] in audio_id_map, \
                    f'{line["id"]} not associated with audio file'
                audio_name = audio_id_map[line['id']]
                if audio_name not in outputs_by_audio:
                    outputs_by_audio[audio_name] = []
                outputs_by_audio[audio_name].append(line)
        return outputs_by_audio

    def _read_all(self) -> List[Any]:
        data = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # skip empty lines
                    data.append(json.loads(line))
        return data

    def num_deleted_tokens(self) -> int:
        """
        Count the number of deleted tokens across all outputs.

        Returns:
            int: Total count of deleted tokens.
        """
        num_deleted_tokens = 0
        for audio, lines in self.outputs_by_audio.items():
            for line in lines:
                if len(line['deleted_tokens']) > 0:
                    num_deleted_tokens += len(
                        text_items(
                            self.detokenizer(line['deleted_tokens']),
                            latency_unit=self.latency_unit))
        return num_deleted_tokens

    def final_outputs_and_latencies(self) -> Dict[str, OutputWithDelays]:
        """
        Compute the final outputs and their associated delays.

        Retranslated (overridden) tokens are excluded from the output and from the delays. When a
        word is partially updated (e.g., only the last subword is updated), the last update latency
        is considered.

        Returns:
            Dict[str, OutputWithDelays]: Mapping of audio file → output with delays.
        """
        outputs: OrderedDict[str, OutputWithDelays] = OrderedDict()
        for audio, lines in self.outputs_by_audio.items():
            tokens = []
            current_output = None
            for line in lines:
                line_delay = line['total_audio_processed']
                line_comp_aware_delay = line['total_audio_processed'] + line['computation_time']
                # remove tokens from previous generation
                if len(line['deleted_tokens']) > 0:
                    assert line['deleted_tokens'] == tokens[-len(line['deleted_tokens']):]
                    tokens = tokens[:-len(line['deleted_tokens'])]
                    # update the current output by removing text and corresponding delays
                    new_output = OutputWithDelays(
                        self.detokenizer(tokens),
                        current_output.ideal_delays,
                        current_output.computational_aware_delays)
                    removed_tokens = current_output.text_len(self.latency_unit) - \
                        new_output.text_len(self.latency_unit)
                    if removed_tokens > 0:
                        new_output.ideal_delays = new_output.ideal_delays[:-removed_tokens]
                        new_output.computational_aware_delays = \
                            new_output.computational_aware_delays[:-removed_tokens]
                    # if the latency unit is `word` and part of the last word has been deleted
                    # we update the latency of the last word
                    if self.latency_unit == "word":
                        previous_ending_word_idx = new_output.text_len("word") - 1
                        if previous_ending_word_idx >= 0:
                            ending_word_before_update = current_output.text_items("word")[
                                previous_ending_word_idx]
                            if ending_word_before_update != new_output.last_word():
                                new_output.ideal_delays[-1] = line_delay
                                new_output.computational_aware_delays[-1] = line_comp_aware_delay
                    current_output = new_output

                # add newly generated tokens
                tokens.extend(line['generated_tokens'])
                # for the first line, we initialize the OutputWithDelays with the partial text and
                # assigning the ideal delay anc computational-aware one to all its units
                if current_output is None:
                    current_output = OutputWithDelays(self.detokenizer(tokens), [], [])
                    num_units = current_output.text_len(self.latency_unit)
                    current_output.ideal_delays = [line_delay] * num_units
                    current_output.computational_aware_delays = [line_comp_aware_delay] * num_units
                else:
                    # update the current output by adding corresponding delays
                    new_output = OutputWithDelays(
                        self.detokenizer(tokens),
                        current_output.ideal_delays,
                        current_output.computational_aware_delays)
                    added_units = new_output.text_len(self.latency_unit) - \
                        current_output.text_len(self.latency_unit)
                    if added_units > 0:
                        new_output.ideal_delays.extend([line_delay] * added_units)
                        new_output.computational_aware_delays.extend(
                            [line_comp_aware_delay] * added_units)
                    # if the latency unit is `word` and part of the last word has been updated
                    # we update the latency of the last word
                    if self.latency_unit == "word":
                        previous_ending_word_idx = current_output.text_len("word") - 1
                        if previous_ending_word_idx >= 0:
                            previous_ending_word_after_update = new_output.text_items("word")[
                                previous_ending_word_idx]
                            if previous_ending_word_after_update != current_output.last_word():
                                new_output.ideal_delays[previous_ending_word_idx] = line_delay
                                new_output.computational_aware_delays[previous_ending_word_idx] = \
                                    line_comp_aware_delay
                    current_output = new_output
            outputs[audio] = current_output
        return outputs

    def final_outputs(self) -> Dict[str, str]:
        """
        Returns the final outputs for each audio.

        Overridden tokens in retranslation are not included in the output, which is the final
        string obtained at the end of the audio file.

        Returns:
            Dict[str, str]: Mapping of audio file → final text.
        """
        outputs: OrderedDict[str, str] = OrderedDict()
        for audio, outputs_with_latency in self.final_outputs_and_latencies().items():
            outputs[audio] = outputs_with_latency.final_text
        return outputs


class ReferencesReader:
    """
    Reads plain-text reference files. Each file corresponds to a single audio.

    Args:
        reference_files (List[str]): Paths to reference files.
    """
    def __init__(self, reference_files: List[str]):
        self.references = self._read_all(reference_files)

    @staticmethod
    def _read_all(references: List[str]) -> Dict[str, List[str]]:
        reference_by_file = OrderedDict()
        for reference in references:
            with open(reference, 'r', encoding='utf-8') as f:
                reference_by_file[Path(reference).stem] = [line.strip() for line in f.readlines()]
        return reference_by_file

    def get_reference_texts(self) -> Dict[str, List[str]]:
        """
        Get the references grouped by file.

        Returns:
           Dict[str, List[str]]: Mapping of file stem → list of reference sentences.
        """
        return self.references


class YamlReferenceReader:
    """
    Reads references aligned with audio definitions.

    The audio definition is a YAML file where each entry describes a segment with its start and
    duration. The reference file contains one sentence per line, where each lines is associated
    with the corresponding segment in the audio definition file.

    Args:
        audio_definition (str): Path to YAML file with segment definitions.
        reference (str): Path to text file with reference sentences.
    """
    def __init__(self, audio_definition: str, reference: str):
        self.references = self._read_all(audio_definition, reference)

    @staticmethod
    def _read_all(
            audio_definition: str, reference: str) -> Dict[str, List[ReferenceSentenceDefinition]]:
        reference_by_file = OrderedDict()
        with open(audio_definition) as f:
            sentence_definitions = yaml.load(f, Loader=yaml.FullLoader)
        with open(reference) as f:
            sentences = f.readlines()
        assert len(sentence_definitions) == len(sentences), \
            f"Number of reference sentences ({len(sentences)}) and sentence definitions " \
            f"({len(sentence_definitions)}) should be the same."
        for sentence, definition in zip(sentences, sentence_definitions):
            wav_name = Path(definition["wav"]).stem
            if wav_name not in reference_by_file:
                reference_by_file[wav_name] = []
            reference_by_file[wav_name].append(ReferenceSentenceDefinition(
                sentence.strip(), definition["offset"], definition["duration"]))
        return reference_by_file

    def get_reference_texts(self) -> Dict[str, List[str]]:
        """
        Get the references grouped by file.

        Returns:
           Dict[str, List[str]]: Mapping of file stem → list of reference sentences.
        """
        return OrderedDict({
            name: [sentence_def.content for sentence_def in list_sentences]
            for name, list_sentences in self.references.items()})
