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
from dataclasses import dataclass
from typing import List, Callable


@dataclass
class IncrementalOutput:
    """
    Represents the incremental output of a speech processor for a single
    processed chunk of audio.

    Attributes:
        new_tokens (List[str]): List of newly generated tokens in this chunk.
        new_string (str): Concatenated string representation of the new tokens.
        deleted_tokens (List[str]): List of tokens that were deleted/overwritten.
        deleted_string (str): Concatenated string representation of the deleted tokens.
    """
    new_tokens: List[str]
    new_string: str
    deleted_tokens: List[str]
    deleted_string: str

    def strings_to_json(self) -> str:
        """
        Serialize the incremental output to a JSON string.

        Returns:
            str: A JSON string containing the newly generated and the deleted text.
        """
        return json.dumps({"new": self.new_string, "deleted": self.deleted_string})


def merge_incremental_outputs(
        outputs: List[IncrementalOutput],
        tokens_to_string: Callable[[List[str]], str]) -> IncrementalOutput:
    """
    Merge the incremental outputs passed as input into a single incremental output.
    The outputs must be sorted in cronological order.

    Args:
        outputs (List[IncrementalOutput]): List of incremental outputs to be merged.
        tokens_to_string (Callable[[List[str]], str]): A function that takes a list of tokens and
            returns a string that contains the detokenized text.
    """
    if len(outputs) == 1:
        return outputs[0]
    if len(outputs) == 0:
        return IncrementalOutput([], "", [], "")

    current_output_tokens = outputs[0].new_tokens
    current_output_deleted_tokens = outputs[0].deleted_tokens
    for output in outputs[1:]:
        num_deleted_tokens = len(output.deleted_tokens)
        if num_deleted_tokens > 0:
            if num_deleted_tokens < len(current_output_tokens):
                assert output.deleted_tokens == current_output_tokens[-num_deleted_tokens:]
                current_output_tokens = current_output_tokens[:-num_deleted_tokens]
            else:
                # we are deleting more than it was generated so far, so extra deleted tokens
                # should be included
                extra_deleted_tokens = output.deleted_tokens[:-len(current_output_tokens)]
                current_output_deleted_tokens = \
                    extra_deleted_tokens + current_output_deleted_tokens
                current_output_tokens = []
        current_output_tokens += output.new_tokens

    return IncrementalOutput(
        current_output_tokens,
        tokens_to_string(current_output_tokens),
        current_output_deleted_tokens,
        tokens_to_string(current_output_deleted_tokens))
