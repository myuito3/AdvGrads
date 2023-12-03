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

"""Base class for defense methods against adversarial attacks."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type

from torch import Tensor

from advgrads.configs.base_config import InstantiateConfig


@dataclass
class DefenseConfig(InstantiateConfig):
    """The base configuration class for defense methods."""

    _target: Type = field(default_factory=lambda: Defense)
    """Target class to instantiate."""


class Defense:
    """The base class for defense methods.

    Args:
        config: Configuration for defense methods.
    """

    config: DefenseConfig

    def __init__(self, config: DefenseConfig, **kwargs) -> None:
        self.config = config

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.run_defense(*args, **kwargs)

    @abstractmethod
    def run_defense(self, x: Tensor, **kwargs) -> Tensor:
        """Return x with defensive processing applied.

        Args:
            x: Image to be applied to the defensive process.
        """
        raise NotImplementedError
