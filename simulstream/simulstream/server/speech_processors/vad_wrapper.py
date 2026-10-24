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

from types import SimpleNamespace
from typing import List

import numpy as np
from silero_vad import load_silero_vad, VADIterator

from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor, \
    speech_processor_class_load
from simulstream.server.speech_processors.incremental_output import merge_incremental_outputs, \
    IncrementalOutput


class VADWrapperSpeechProcessor(SpeechProcessor):
    """
    A speech processor that integrates **Voice Activity Detection (VAD)** to filter and split
    continuous audio streams into meaningful speech chunks before processing them with an
    underlying speech processor.

    This class wraps a :class:`SpeechProcessor` implementation (defined by in the configuration via
    the attribute `base_speech_processor_class`) with a Silero VAD-based iterator that detects the
    start and end of speech segments. Audio outside of speech is ignored, and each detected segment
    is passed to the underlying speech processor.

    Args:
        config (SimpleNamespace): Configuration object. The following attributes are used:

            - **base_speech_processor_class (str)**: full name of the underlying speech processor
              class to use.
            - **vad_threshold (float, optional)**: VAD probability threshold. Default = ``0.5``.
            - **vad_min_silence_duration_ms (int, optional)**: Minimum silence duration
              (milliseconds) to consider the end of a speech segment. Default = ``100``.
            - **vad_speech_pad_ms (int, optional)**: Padding (milliseconds) to include before and
              after detected speech. Default = ``30``.
            - **min_speech_size (int, optional)**: Minimum segment size in seconds; shorter
              segments are ignored. Default = ``1``.
            - Any additional attributes required by the subclass :py:attr:`speech_processor_class`.
    """

    @classmethod
    def speech_processor_class(cls, config: SimpleNamespace) -> type[SpeechProcessor]:
        return speech_processor_class_load(config.base_speech_processor_class)

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        super().load_model(config)
        if not hasattr(cls, "vad_model") or cls.vad_model is None:
            cls.vad_model = load_silero_vad()
        cls.speech_processor_class(config).load_model(config)

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.speech_processor = self.speech_processor_class(self.config)(self.config)
        self.min_speech_size = getattr(self.config, "min_speech_size", 1) * SAMPLE_RATE
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=getattr(self.config, "vad_threshold", 0.5),
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=getattr(self.config, "vad_min_silence_duration_ms", 100),
            speech_pad_ms=getattr(self.config, "vad_speech_pad_ms", 30),
        )
        self.residual_prev_audio = None
        self.speech_buffer = None
        self.audio_cursor_position = 0
        self.in_speech = False
        assert SAMPLE_RATE == 16000, \
            "SileroHFSlidingWindowRetranslator supports only 16kHz sampling rate"
        self.window_size_samples = 512  # assuming 16kHz
        self.previous_audio_chunk = None  # needed as VAD uses padding before start

    def clear(self) -> None:
        super().clear()
        self.residual_prev_audio = None
        self.speech_buffer = None
        self.audio_cursor_position = 0
        self.in_speech = False
        self.previous_audio_chunk = None
        self.vad_iterator.reset_states()
        self.speech_processor.clear()

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        if self.residual_prev_audio is not None:
            waveform = np.concatenate((self.residual_prev_audio, waveform))
            self.residual_prev_audio = None
        # we can have more than one generate if there are multiple speech segments in the current
        # chunk
        outputs = []

        for i in range(0, len(waveform), self.window_size_samples):
            chunk = waveform[i: i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                # process tailing audio with the next waveform chunk
                self.residual_prev_audio = chunk
                break
            speech_dict = self.vad_iterator(chunk, return_seconds=False)
            if speech_dict:
                # if a VAD event happens, update the status accordingly
                assert not ('start' in speech_dict and 'end' in speech_dict)
                if 'start' in speech_dict:
                    assert not self.in_speech, \
                        "Cannot start a new segment when current one is being processed. " \
                        "This means there is a bug in the implementation."
                    start_offset = self.audio_cursor_position - len(self.previous_audio_chunk)
                    chunk_start_position = speech_dict['start'] - start_offset
                    assert chunk_start_position >= 0
                    self.speech_buffer = np.concatenate(
                        (self.previous_audio_chunk, chunk))[chunk_start_position:]
                    self.in_speech = True
                if 'end' in speech_dict:
                    assert self.in_speech, \
                        "Cannot end a segment when no current segment is being processed. " \
                        "This means there is a bug in the implementation."
                    speech_buffer_len = len(self.speech_buffer) \
                        if self.speech_buffer is not None else 0
                    speech_buffer_offset = self.audio_cursor_position - speech_buffer_len
                    chunk_end_position = speech_dict['end'] - speech_buffer_offset
                    # In case we already processed more audio than needed (i.e., we already
                    # processed beyond the end by VAD, we skip further processing; otherwise, we
                    # process the remaining unhandled audio).
                    # We can process more audio after the end as the VAD takes ~100ms (see the
                    # min_silence_duration_ms parameter of the VAD) to emit the end signal, so if
                    # we already processed the partial speech buffer (see min_speech_size in this
                    # processor), we may have processed up to extra 100ms.
                    if chunk_end_position >= 0:
                        self.append_to_speech_buffer(chunk)
                        self.speech_buffer = self.speech_buffer[:chunk_end_position]
                        outputs.append(self.speech_processor.process_chunk(self.speech_buffer))
                    self.in_speech = False
                    self.speech_buffer = None
                    outputs.append(self.speech_processor.end_of_stream())
                    # reset history at the end of a segment
                    if hasattr(self.speech_processor, 'text_history'):
                        self.speech_processor.text_history = None
                    if hasattr(self.speech_processor, 'audio_history'):
                        self.speech_processor.audio_history = None
            else:
                # if no VAD event happens, we just ignore the audio if we are outside speech and
                # update the buffer in case we are in speech
                if self.in_speech:
                    self.append_to_speech_buffer(chunk)
            # update cursor position
            self.audio_cursor_position += self.window_size_samples
            self.previous_audio_chunk = chunk

        if self.in_speech and len(self.speech_buffer) > self.min_speech_size:
            outputs.append(self.speech_processor.process_chunk(self.speech_buffer))
            self.speech_buffer = None

        return merge_incremental_outputs(outputs, self.tokens_to_string)

    def append_to_speech_buffer(self, audio_chunk: np.float32) -> None:
        if self.speech_buffer is None:
            self.speech_buffer = audio_chunk
        else:
            self.speech_buffer = np.concatenate((self.speech_buffer, audio_chunk))

    def set_source_language(self, language: str) -> None:
        self.speech_processor.set_source_language(language)

    def set_target_language(self, language: str) -> None:
        self.speech_processor.set_target_language(language)

    def tokens_to_string(self, tokens: List[str]) -> str:
        return self.speech_processor.tokens_to_string(tokens)

    def end_of_stream(self) -> IncrementalOutput:
        return self.speech_processor.end_of_stream()
