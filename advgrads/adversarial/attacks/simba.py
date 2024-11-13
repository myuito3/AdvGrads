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

"""The implementation of the Simple Black-box Attack (SimBA) attack.

Paper: Simple Black-box Adversarial Attacks
Url: https://arxiv.org/abs/1905.07121

Original code is referenced from https://github.com/cg563/simple-blackbox-attack
Note that this code is simply an extension of 20-line implementation of SimBA to batch
processing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import Attack, AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.utils.losses import MarginLoss
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


@dataclass
class SimBAAttackConfig(AttackConfig):
    """The configuration class for the SimBA attack."""

    _target: Type = field(default_factory=lambda: SimBAAttack)
    """Target class to instantiate."""
    lr: float = 0.2
    """Step size per iteration."""
    freq_dims: int = 14
    """Dimensionality of 2D frequency space."""


class SimBAAttack(Attack):
    """The class of the SimBA attack.

    Args:
        config: The SimBA attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: SimBAAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_2"]

    def __init__(self, config: SimBAAttackConfig) -> None:
        super().__init__(config)

        if self.eps > 0.0:
            raise ValueError(
                "SimBA is a minimum-norm attack, not a norm-constrained attack."
            )
        if self.max_iters > 0:
            raise ValueError(
                "The maximum number of queries for SimBA is controlled by the "
                "freq_dims parameter in the config."
            )

        self.loss = (
            nn.CrossEntropyLoss(reduction="none")
            if self.targeted
            else MarginLoss(targeted=self.targeted)
        )
        self.margin = MarginLoss(self.targeted)

    def reset_data(self, *args: Tuple[Tensor, ...]) -> None:
        """Register data to be used for attack.

        Args:
            args: Tensor data such as images and loss values.
        """
        assert all(
            args[0].shape[0] == arg.shape[0] for arg in args
        ), "Size mismatch between tensors."
        self.all_args = args

    def get_data(self, indices: Tensor) -> Tuple[Tensor, ...]:
        """Returns only the elements specified by indices from registered data.

        Args:
            indices: Indices of data to be extracted.
        """
        return (arg[indices] for arg in self.all_args)

    def update_data(self, indices: Tensor, *args: Tuple[Tensor, ...]) -> None:
        """Update the data at specified indices in registered data with new data.

        Args:
            indices: Indices of data to be replaced.
            args: New tensor data.
        """
        for arg, new_arg in zip(self.all_args, args):
            arg[indices] = new_arg

    @torch.no_grad()
    def step_single(self, idx_to_fool: Tensor, diffs: Tensor, model: Model) -> Tensor:
        """Perform one step of attack with given additional perturbations.

        Args:
            idx_to_fool: Indices of data to be used to attack.
            diffs: Additional perturbations.
            model: Model to be attacked.
        """
        x_best, y, loss_min, margin_min = self.get_data(idx_to_fool)

        # Generate candidates for new adversarial examples.
        x_new = torch.clamp(x_best + diffs, min=self.min_val, max=self.max_val)
        logits = model(x_new)
        loss = self.loss(logits, y)
        margin = self.margin(logits, y)

        # Update current loss values and adversarial examples.
        idx_improved = loss < loss_min
        loss_min = idx_improved * loss + ~idx_improved * loss_min
        margin_min = idx_improved * margin + ~idx_improved * margin_min

        _idx_improved = torch.reshape(idx_improved, [-1, *[1] * len(x_best.shape[:-1])])
        x_best = _idx_improved * x_new + ~_idx_improved * x_best

        self.update_data(idx_to_fool, x_best, y, loss_min, margin_min)
        return idx_improved

    @torch.no_grad()
    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        c, h, w = x.shape[1:]
        n_queries = torch.zeros((x.shape[0]), dtype=torch.int16).to(x.device)

        x_best = x.clone()
        logits = model(x_best)
        loss_min = self.loss(logits, y)
        margin_min = self.margin(logits, y)
        n_queries += 1

        # Determine index of pixels to be perturbed at random for each image.
        n_dims = c * self.config.freq_dims * self.config.freq_dims
        idx_pixels = torch.zeros((x.shape[0], n_dims), device=x.device).long()
        for i in range(x.shape[0]):
            idx_pixels[i, ...] = torch.randperm(c * h * w)[:n_dims]

        self.reset_data(x_best, y, loss_min, margin_min)

        for i_iter in range(n_dims):
            idx_to_fool = torch.atleast_1d((margin_min > 0.0).nonzero().squeeze())
            if len(idx_to_fool) == 0:
                break

            # Try negative direction.
            diffs = torch.zeros((len(idx_to_fool), c * h * w), device=x.device)
            u = torch.arange(len(idx_to_fool))
            diffs[u, idx_pixels[idx_to_fool, i_iter]] = -1 * self.config.lr
            diffs = diffs.view(-1, *x.shape[1:])

            idx_improved = self.step_single(idx_to_fool, diffs, model)
            n_queries[idx_to_fool] += 1

            # Try positive direction for samples that failed to update loss by trying
            # negative direction.
            idx_failed = torch.nonzero(~idx_improved).squeeze()
            idx_to_fool = torch.atleast_1d(idx_to_fool[idx_failed])
            if len(idx_to_fool) == 0:
                continue

            diffs = torch.zeros((len(idx_to_fool), c * h * w), device=x.device)
            u = torch.arange(len(idx_to_fool))
            diffs[u, idx_pixels[idx_to_fool, i_iter]] = self.config.lr
            diffs = diffs.view(-1, *x.shape[1:])

            _ = self.step_single(idx_to_fool, diffs, model)
            n_queries[idx_to_fool] += 1

        x_best, _, _, _ = self.get_data(torch.arange(x.shape[0]))
        return {ResultHeadNames.X_ADV: x_best, ResultHeadNames.QUERIES: n_queries}

    def get_metrics_dict(
        self, outputs: Dict[ResultHeadNames, Tensor], batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        metrics_dict = {}
        succeed = outputs[ResultHeadNames.SUCCEED]

        # query
        queries_succeed = outputs[ResultHeadNames.QUERIES][succeed]
        metrics_dict[ResultHeadNames.QUERIES_SUCCEED] = queries_succeed

        # perturbation norm
        l2_norm_succeed = torch.norm(
            outputs[ResultHeadNames.X_ADV] - batch["images"], p=2, dim=[1, 2, 3]
        )[succeed]
        metrics_dict["l2_norm"] = l2_norm_succeed

        return metrics_dict
