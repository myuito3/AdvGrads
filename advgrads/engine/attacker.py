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

"""Code to perform adversarial attacks."""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Type, Union

import numpy as np
import torch

from advgrads.adversarial import get_attack_config_class, get_defense_config_class
from advgrads.adversarial.attacks.base_attack import AttackConfig, Attack
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.adversarial.defenses.input_transform.base_defense import DefenseConfig
from advgrads.configs.experiment_config import ExperimentConfig
from advgrads.data import get_dataset_class
from advgrads.data.utils import index_samplers
from advgrads.data.utils.data_utils import get_dataloader
from advgrads.models import get_model_config_class
from advgrads.models.base_model import ModelConfig, Model
from advgrads.utils.metrics import SuccessRateMeter, QueryMeter
from advgrads.utils.rich_utils import console_log, console_print, Panel


def _set_random_seed(seed: int) -> None:
    """Set randomness seed in torch and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class AttackerConfig(ExperimentConfig):
    """Configuration for attacker instantiation."""

    _target: Type = field(default_factory=lambda: Attacker)
    """Target class to instantiate."""
    device: Union[str, torch.device] = "cpu"
    """The device to perform the attack on."""


class Attacker:
    """The attacker class.

    Args:
        config: Configuration for attacker instantiation.
    """

    def __init__(self, config: AttackerConfig) -> None:
        self.config = config

        model_config: ModelConfig = get_model_config_class(config.model)()
        self.model: Model = model_config.setup()
        self.model.load()
        self.model.to(self.device)

        self.defense = None
        if config.thirdparty_defense is not None:
            defense_config: DefenseConfig = get_defense_config_class(
                config.thirdparty_defense
            )()
            self.defense = defense_config.setup()

        if config.data == "imagenet":
            image_indices = (
                index_samplers.get_random(config.num_images, population=50000)
                if config.num_images is not None
                else None
            )
            dataset = get_dataset_class(config.data)(
                transform=self.model.get_transform(), indices_to_use=image_indices
            )
        else:
            image_indices = (
                index_samplers.get_random(config.num_images)
                if config.num_images is not None
                else None
            )
            dataset = get_dataset_class(config.data)(indices_to_use=image_indices)

        self.dataset = dataset
        self.dataloader = get_dataloader(dataset, batch_size=config.batch_size)

    @property
    def device(self) -> str:
        return self.config.device

    def get_attack_outputs(self, attack: Attack) -> Dict[ResultHeadNames, Any]:
        """Returns the result of executing an attack.

        Args:
            attack: The instance of an attack method to be executed.
        """
        success_rate_meter = SuccessRateMeter()
        query_meter = QueryMeter()

        for images, labels in self.dataloader:
            if attack.targeted:
                # Currently we use gt+1 as the target label.
                labels = (labels + 1) % self.dataset.num_classes

            batch = {"images": images.to(self.device), "labels": labels.to(self.device)}
            attack_outputs = attack(batch, self.model, thirdparty_defense=self.defense)
            success_rate_meter.update(
                attack_outputs[ResultHeadNames.NUM_SUCCEED], len(images)
            )

            metrics_dict = attack.get_metrics_dict(attack_outputs, batch)
            if ResultHeadNames.QUERIES_SUCCEED in metrics_dict.keys():
                query_meter.update(metrics_dict[ResultHeadNames.QUERIES_SUCCEED])

            console_log(str(success_rate_meter) + str(query_meter))

        console_print(
            Panel(
                f"Method: {attack.method} "
                + str(success_rate_meter)
                + str(query_meter),
                title="[bold][green]:tada: Attack Finished :tada:[/bold]",
                expand=False,
            )
        )

        outputs = {ResultHeadNames.SUCCESS_RATE: success_rate_meter.get_success_rate()}
        outputs[ResultHeadNames.MEAN_QUERY] = query_meter.get_mean()
        outputs[ResultHeadNames.MEDIAN_QUERY] = query_meter.get_median()

        return outputs

    def run(self) -> None:
        """Execute all the methods in the attack list and output the results to a file
        with the experimental settings.
        """
        for attack_dict in self.config.attacks:
            _set_random_seed(self.config.seed)

            attack_config: AttackConfig = get_attack_config_class(
                attack_dict["method"]
            )()
            attack_config.__dict__.update(attack_dict)
            attack = attack_config.setup()

            attack_outputs = self.get_attack_outputs(attack)

            result_config = deepcopy(self.config)
            result_config.__dict__.update(attack_config.__dict__)
            result_config.__dict__.update(attack_outputs)
            result_config.save_config()
