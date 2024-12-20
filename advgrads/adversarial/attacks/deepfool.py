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

"""The implementation of the DeepFool attack.

Paper: DeepFool: a simple and accurate method to fool deep neural networks
Url: https://arxiv.org/abs/1511.04599
"""

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import Attack, AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.utils.types import AttackOutputs
from advgrads.models.base_model import Model
from advgrads.utils.printing import print_as_warning


@dataclass
class DeepFoolAttackConfig(AttackConfig):
    """The configuration class for the DeepFool attack."""

    _target: Type = field(default_factory=lambda: DeepFoolAttack)
    """Target class to instantiate."""
    eta: float = 0.02
    """Parameter for crossing classification boundaries."""


class DeepFoolAttack(Attack):
    """The class of the DeepFool attack.

    Args:
        config: The DeepFool attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: DeepFoolAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_2"]

    def __init__(self, config: DeepFoolAttackConfig) -> None:
        super().__init__(config)

        if self.targeted:
            print_as_warning(f"{self.method} does not support targeted attack.")
            self.config.targeted = False
        if self.eps > 0.0:
            print_as_warning(
                f"{self.method} is a minimum-norm attack, not a norm-constrained attack."
            )
            self.config.eps = 0.0

    def deepfool_single(self, x: Tensor, y: Tensor, model: Model) -> Tensor:
        """Single adversarial perturbation generation by DeepFool."""
        x_adv = x.clone().requires_grad_(True)
        logits = model(x_adv)
        pred = torch.argmax(logits, dim=-1)

        delta = torch.zeros_like(x, device=x.device)
        num_classes = logits.shape[-1]

        for _ in range(self.max_iters):
            if pred != y:
                break

            # Find the best direction of the step.
            grad_0 = torch.autograd.grad(logits[0, pred], [x_adv], retain_graph=True)[
                0
            ].detach()
            w_best = None
            pert_best = torch.inf

            for k in range(num_classes):
                if k == pred:
                    continue

                grad_1 = torch.autograd.grad(logits[0, k], [x_adv], retain_graph=True)[
                    0
                ].detach()
                w_k = grad_1 - grad_0
                f_k = logits[0, k] - logits[0, pred]
                pert_k = (torch.abs(f_k) + 1e-5) / torch.norm(
                    w_k.flatten(), p=2, dim=-1
                )

                if pert_k < pert_best:
                    w_best = w_k
                    pert_best = pert_k

            # Accumulate updates in the delta.
            r_i = (
                (pert_best + 1e-4) * w_best / torch.norm(w_best.flatten(), p=2, dim=-1)
            )
            delta += r_i.clone()

            x_adv = torch.clamp(
                x + delta, min=self.min_val, max=self.max_val
            ).requires_grad_(True)
            logits = model(x_adv)
            pred = torch.argmax(logits, dim=-1)

        return delta

    def run_attack(self, x: Tensor, y: Tensor, model: Model) -> AttackOutputs:
        x_adv = torch.zeros_like(x, device=x.device)

        for i_img in range(x.shape[0]):
            model.zero_grad()

            x_i = x[i_img : i_img + 1]
            y_i = y[i_img : i_img + 1]

            delta = self.deepfool_single(x_i, y_i, model)
            x_adv[i_img] = torch.clamp(
                (1 + self.config.eta) * delta + x_i, min=self.min_val, max=self.max_val
            )

        return AttackOutputs(x_adv=x_adv)

    def get_metrics_dict(
        self, outputs: AttackOutputs, x: Tensor, y: Tensor, succeed: Tensor, **kwargs
    ) -> Dict[str, Tensor]:
        metrics_dict = {}

        # perturbation norm
        l2_norm_succeed = torch.norm(outputs.x_adv - x, p=2, dim=[1, 2, 3])[succeed]
        metrics_dict["l2_norm"] = l2_norm_succeed
        return metrics_dict
