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

import unittest

from simulstream.config import yaml_config
from simulstream.metrics.readers import LogReader
from uts.utils import ASSETS_DIR, CONFIGS_DIR


class LogReaderTestCase(unittest.TestCase):
    def test_log_reader(self):
        metrics_file = ASSETS_DIR / "test_metrics.jsonl"
        config_file = CONFIGS_DIR / "seamless_sliding_window_retranslation.yaml"
        config = yaml_config(config_file)
        log_reader = LogReader(config, metrics_file)
        output_with_latency_by_wav = log_reader.final_outputs_and_latencies()
        self.assertEqual(len(output_with_latency_by_wav), 1)
        output_with_latency = next(iter(output_with_latency_by_wav.values()))
        self.assertEqual(output_with_latency.ideal_delays[:8], ([2] * 7) + [4])
        self.assertEqual(
            output_with_latency.text_len("word"), len(output_with_latency.ideal_delays))
        self.assertEqual(
            output_with_latency.text_len("word"),
            len(output_with_latency.computational_aware_delays))

    def test_all_text_deleted(self):
        metrics_file = ASSETS_DIR / "test_all_text_deleted.jsonl"
        config_file = CONFIGS_DIR / "fama_hf_sliding_window_retranslation.yaml"
        config = yaml_config(config_file)
        log_reader = LogReader(config, metrics_file)
        output_with_latency_by_wav = log_reader.final_outputs_and_latencies()
        self.assertEqual(len(output_with_latency_by_wav), 1)
        output_with_latency = next(iter(output_with_latency_by_wav.values()))
        self.assertEqual(len(output_with_latency.ideal_delays), 6)


if __name__ == '__main__':
    unittest.main()
