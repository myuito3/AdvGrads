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
import random
from pathlib import Path

import numpy as np
import torch

from advgrads.adversarial import get_attack_config_class, get_defense_config_class
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.configs.experiment_config import ExperimentConfig, ResultConfig
from advgrads.data import get_dataset_class
from advgrads.data.utils import index_samplers
from advgrads.data.utils.data_utils import get_dataloader
from advgrads.models import get_model_config_class
from advgrads.utils.io import load_from_yaml
from advgrads.utils.metrics import SuccessRateMeter, QueryMeter
from advgrads.utils.rich_utils import console_log, console_print, Panel


def _set_random_seed(seed: int) -> None:
    """Set randomness seed in torch and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@click.command()
@click.option("--load_config", type=str, required=True, help="Config file.")
def main(load_config) -> None:
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig()
    config.__dict__.update(load_from_yaml(Path(load_config)))

    model_config = get_model_config_class(config.model)()
    model = model_config.setup()
    model.load()
    model.to(device)

    if "imagenet" in config.model:
        image_indices = (
            index_samplers.get_random(config.num_images, population=50000)
            if config.num_images is not None
            else None
        )
        dataset = get_dataset_class(config.data)(
            transform=model.get_transform(), indices_to_use=image_indices
        )
    else:
        image_indices = (
            index_samplers.get_arange(config.num_images)
            if config.num_images is not None
            else None
        )
        dataset = get_dataset_class(config.data)(indices_to_use=image_indices)
    dataloader = get_dataloader(dataset, batch_size=config.batch_size)

    defense = None
    if config.thirdparty_defense is not None:
        defense_config = get_defense_config_class(config.thirdparty_defense)()
        defense = defense_config.setup()

    for attack_dict in config.attacks:
        _set_random_seed(config.seed)

        attack_config = get_attack_config_class(attack_dict["method"])()
        attack_config.__dict__.update(attack_dict)
        attack = attack_config.setup()

        success_rate_meter = SuccessRateMeter()
        query_meter = QueryMeter()

        for images, labels in dataloader:
            if attack.targeted:
                # Currently we use gt+1 as the target label.
                labels = (labels + 1) % dataset.num_classes

            images, labels = images.to(device), labels.to(device)
            attack_outputs = attack(images, labels, model, thirdparty_defense=defense)

            if ResultHeadNames.NUM_SUCCEED in attack_outputs.keys():
                success_rate_meter.update(
                    attack_outputs[ResultHeadNames.NUM_SUCCEED], len(images)
                )
            if ResultHeadNames.QUERIES_SUCCEED in attack_outputs.keys():
                query_meter.update(attack_outputs[ResultHeadNames.QUERIES_SUCCEED])

            console_log(str(success_rate_meter) + str(query_meter))

        outputs = {ResultHeadNames.SUCCESS_RATE: success_rate_meter.get_success_rate()}
        outputs[ResultHeadNames.MEAN_QUERY] = query_meter.get_mean()
        outputs[ResultHeadNames.MEDIAN_QUERY] = query_meter.get_median()

        console_print(
            Panel(
                str(success_rate_meter) + str(query_meter),
                title="[bold][green]:tada: Attack Finished :tada:[/bold]",
                expand=False,
            )
        )

        result_config = ResultConfig()
        result_config.__dict__.update(config.__dict__)
        if config.thirdparty_defense is not None:
            result_config.__dict__.update(defense_config.__dict__)
        result_config.__dict__.update(attack_config.__dict__)
        result_config.__dict__.update(outputs)
        result_config.save_config()


if __name__ == "__main__":
    main()
