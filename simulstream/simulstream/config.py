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

import yaml


def _dict_to_object(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_object(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_object(i) for i in d]
    else:
        return d


def yaml_config(path: str) -> SimpleNamespace:
    with open(path, "r") as file:
        return _dict_to_object(yaml.safe_load(file))
