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

"""Init model configs."""

from advgrads.adversarial.defenses.adv_train.trades.trades_mnist import TradesMnistModel
from advgrads.models.base_model import Model
from advgrads.models.pytorch_playground.cifar10_model import PtPgCifar10Model
from advgrads.models.pytorch_playground.mnist_model import PtPgMnistModel


def get_model_class(name: str) -> Model:
    return model_class_dict[name]


model_class_dict = {
    "ptpg_mnist": PtPgMnistModel,
    "ptpg_cifar10": PtPgCifar10Model,
    "trades_mnist": TradesMnistModel,
}
all_model_names = list(model_class_dict.keys())
