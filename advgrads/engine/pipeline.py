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

from typing import Optional, Union

import torch
from rich.panel import Panel
from torch.utils.data import Dataset

from advgrads.adversarial import get_defense
from advgrads.adversarial.attacks.base_attack import Attack, AttackConfig
from advgrads.adversarial.attacks.utils.types import AttackOutputs
from advgrads.adversarial.defenses.input_transform.base_defense import Defense
from advgrads.data import get_dataset
from advgrads.data.utils.data_utils import get_dataloader
from advgrads.models import get_model
from advgrads.models.base_model import Model
from advgrads.utils.metrics import SuccessRateMeter, QueryMeter
from advgrads.utils.misc import set_seed
from advgrads.utils.printing import CONSOLE, print_as_error


class Pipeline:
    """The pipeline class."""

    dataset: Dataset
    model: Model
    defense: Optional[Union[Defense, Model]] = None

    def __init__(
        self,
        dataset_name: str,
        num_images: int,
        batch_size: int,
        model_name: str,
        defense_method: Optional[str] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.defense_method = defense_method
        self.seed = seed
        self.device = device

        self.model = get_model(model_name)
        self.model.load()
        self.model.to(self.device)

        if defense_method is not None:
            self.defense = get_defense(defense_method)

        self.dataset, self.image_indices = get_dataset(
            dataset_name,
            num_images=num_images,
            transform=self.model.get_transform(),
        )
        self.dataloader = get_dataloader(self.dataset, batch_size=batch_size)

    def get_attack_outputs(self, attack: Attack) -> AttackOutputs:
        """Returns the result of executing an attack.

        Args:
            attack: The instance of an attack method to be executed.
        """

        adv_images = []
        success_rate_meter = SuccessRateMeter()
        query_meter = QueryMeter()

        for images, labels in self.dataloader:
            if attack.targeted:
                # Currently we use gt+1 as the target label.
                labels = (labels + 1) % self.dataset.num_classes

            images = images.to(self.device)
            labels = labels.to(self.device)

            attack_outputs = attack(images, labels, self.model)
            adv_images.append(attack_outputs.x_adv.detach())

            # If a defensive method is defined, the process is performed here. This
            # corresponds to Section 5.2 (GRAY BOX: IMAGE TRANSFORMATIONS AT TEST TIME)
            # in the paper of Guo et al [https://arxiv.org/pdf/1711.00117.pdf].
            with torch.no_grad():
                logits = (
                    self.model(self.defense(attack_outputs.x_adv.detach()))
                    if self.defense is not None
                    else self.model(attack_outputs.x_adv.detach())
                )
            preds = torch.argmax(logits, dim=-1)
            succeed = (preds == labels) if attack.targeted else (preds != labels)

            metrics = {
                "pred_labels": preds,
                "succeed": succeed,
                "num_succeed": succeed.sum(),
            }
            metrics_dict = attack.get_metrics_dict(
                attack_outputs, images, labels, **metrics
            )
            success_rate_meter.update(succeed.sum(), len(images))
            if "queries_succeed" in metrics_dict.keys():
                query_meter.update(metrics_dict["queries_succeed"])

            CONSOLE.log(str(success_rate_meter) + str(query_meter))

        adv_images = torch.cat(adv_images, dim=0)

        CONSOLE.print(
            Panel(
                f"Method: {attack.method} "
                + str(success_rate_meter)
                + str(query_meter),
                title="[bold][green]:tada: Attack Finished :tada:[/bold]",
                expand=False,
            )
        )

        outputs = {}

        outputs["success_rate"] = success_rate_meter.get_success_rate()
        outputs["mean_query"] = query_meter.get_mean()
        outputs["median_query"] = query_meter.get_median()

        return outputs, adv_images

    def run(self, attack_config: AttackConfig) -> None:
        """Execute all the methods in the attack list and output the results to a file
        with the experimental settings.
        """
        set_seed(self.seed)

        try:
            attack = attack_config.setup()
            attack_outputs, adv_images = self.get_attack_outputs(attack)
            return attack_outputs, adv_images
        except Exception as e:
            print_as_error(attack_config.method, ":", e)
            return
