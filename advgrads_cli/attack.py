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

"""Code to be called from the command line to perform attacks."""

import click
from pathlib import Path

import torch

from advgrads.engine.attacker import AttackerConfig, Attacker
from advgrads.utils.io import load_from_yaml


@click.command()
@click.option("--load_config", type=str, required=True, help="Config file.")
def main(load_config) -> None:
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AttackerConfig(experiment_name=Path(load_config).stem, device=device)
    config.__dict__.update(load_from_yaml(Path(load_config)))
    attacker: Attacker = config.setup()
    attacker.run()


if __name__ == "__main__":
    main()
