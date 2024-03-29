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

"""Base configs."""

from dataclasses import dataclass
from typing import Any, Type


class BaseConfig:
    """Base config class."""

    def update(self, *args) -> None:
        """Update oneself configs from objects with a dictionary or a dictionary
        itself.
        """
        for _from in args:
            if isinstance(_from, dict):
                self.__dict__.update(_from)
            if hasattr(_from, "__dict__"):
                self.__dict__.update(_from.__dict__)


@dataclass
class InstantiateConfig(BaseConfig):
    """Config class for instantiating an the class specified in the _target
    attribute.
    """

    _target: Type
    """Target class to instantiate."""

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)
