# Copyright 2023 Makoto Yuito. All rights reserved.
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

"""Index samplers for selecting images from a dataset."""

import random
from typing import List


def get_arange(stop: int, start: int = 0) -> List[int]:
    """Returns a list of numbers in the range.

    Args:
        start: Index to start sampling from. Default is 0.
        stop: Index to stop sampling. Returning list does not include this value.
    """
    return list(range(start, stop))


def get_random(num_samples: int, population: int = 10000) -> List[int]:
    """Returns a list of randomly sampled numbers without duplicates.

    Args:
        population: Index population. Random numbers are sampled from 0 to this value.
        num_samples: Number of samples.
    """
    return random.sample(range(population), k=num_samples)
