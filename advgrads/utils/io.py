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

"""Input/output utils."""

from pathlib import Path

import yaml


def load_from_yaml(filename: Path) -> dict:
    """Load a dictionary from a YAML filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".yaml"
    with open(filename, mode="r", encoding="UTF-8") as file:
        return yaml.safe_load(file)


def write_to_yaml(filename: Path, content: dict) -> None:
    """Write data to a YAML file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".yaml"
    with open(filename, mode="w", encoding="UTF-8") as file:
        yaml.dump(content, file, sort_keys=False)
