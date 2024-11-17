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

import numpy as np
import torch
from torch import Tensor

from advgrads.adversarial import get_attack_config
from advgrads.adversarial.attacks.base_attack import AttackConfig
from advgrads.engine.pipeline import Pipeline
from advgrads.utils.io import load_from_yaml


def save_results(
    attack_config: AttackConfig,
    attack_outputs: dict,
    adv_images: Tensor,
    save_adv_image: bool = True,
):
    attack_config.update(attack_outputs)
    attack_config.save_config()

    if save_adv_image:
        save_path = attack_config.get_base_dir() / "results.npz"
        adv_images = adv_images.cpu().numpy()
        np.savez(save_path, adv_images=adv_images)


@click.command()
@click.option("--load_config", type=str, required=True, help="Config file.")
def main(load_config) -> None:
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yaml_configs = load_from_yaml(Path(load_config))
    attacks = yaml_configs.pop("attacks")

    pipeline = Pipeline(device=device, **yaml_configs)

    for attack_dict in attacks:
        attack_dict.update(yaml_configs)
        attack_config = get_attack_config(attack_dict)
        attack_outputs, adv_images = pipeline.run(attack_config)

        save_results(attack_config, attack_outputs, adv_images)


if __name__ == "__main__":
    main()
