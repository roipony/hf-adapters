# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""to_default_device_layout must not touch tensors that are not on Spyre."""

import torch

from hf_adapters.hf_common import to_default_device_layout


def test_cpu_tensor_is_returned_as_is():
    x = torch.randn(1, 8, 64, dtype=torch.float16)
    assert to_default_device_layout(x) is x
