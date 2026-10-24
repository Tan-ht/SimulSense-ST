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

import logging


METRICS_LOGGER = logging.getLogger('fbk_fairseq.simultaneous.metrics')
METRICS_LOGGER.propagate = False


def setup_metrics_logger(metrics_config):
    if metrics_config.enabled:
        fh = logging.FileHandler(metrics_config.filename)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)

        # Clear existing handlers (if any) and set new one
        METRICS_LOGGER.handlers.clear()
        METRICS_LOGGER.addHandler(fh)
    else:
        METRICS_LOGGER.disabled = True
