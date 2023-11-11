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

"""Config used for running an experiment."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from advgrads.utils.io import write_to_yaml


@dataclass
class ExperimentConfig:
    """Full config contents for running an experiment."""

    data: Optional[str] = None
    """Name of the dataset."""
    model: Optional[str] = None
    """Name of the model."""
    checkpoint_path: Optional[str] = None
    """Path to the checkpoint to be loaded into the model."""
    attacks: Optional[List[dict]] = None
    """List of attack parameters."""
    seed: Optional[int] = None
    """Seed of random number."""
    num_images: Optional[int] = None
    """Number of images used for the attack."""
    batch_size: Optional[int] = None
    """Number of images per batch."""
    thirdparty_defense: Optional[str] = None
    """Name of thirdparty defense method."""


@dataclass
class ResultConfig(ExperimentConfig):
    """Result config."""

    output_dir: Path = Path("outputs")
    """Output directory to save the result of each attack."""
    method: Optional[str] = None
    """Name of attack method."""

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths."""
        return Path(f"{self.output_dir}/{self.method}")

    def save_config(self) -> None:
        """Save config to base directory."""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)

        delattr(self, "attacks")
        output_path = base_dir / f"{self.model}.yaml"
        write_to_yaml(output_path, self)
