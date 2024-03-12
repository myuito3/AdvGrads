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

from typing import Type

from advgrads.adversarial.defenses.adv_train.trades.trades_mnist import (
    TradesMnistModelConfig,
)
from advgrads.models.base_model import ModelConfig, Model
from advgrads.models.imagenet.inception import InceptionV3ImagenetModelConfig
from advgrads.models.imagenet.resnet import Resnet50ImagenetModelConfig
from advgrads.models.imagenet.vgg import (
    Vgg16ImagenetModelConfig,
    Vgg16bnImagenetModelConfig,
)
from advgrads.models.pytorch_playground.cifar10_model import PtPgCifar10ModelConfig
from advgrads.models.pytorch_playground.mnist_model import PtPgMnistModelConfig


def get_model_config_class(name: str) -> Type[Model]:
    assert name in all_model_names, f"Model named '{name}' not found."
    return model_config_class_dict[name]


def get_model(name: str, **kwargs) -> Model:
    model_config: ModelConfig = get_model_config_class(name)(**kwargs)
    model = model_config.setup()
    return model


model_config_class_dict = {
    "ptpg-mnist": PtPgMnistModelConfig,
    "ptpg-cifar10": PtPgCifar10ModelConfig,
    "trades-mnist": TradesMnistModelConfig,
    "inceptionv3-imagenet": InceptionV3ImagenetModelConfig,
    "resnet50-imagenet": Resnet50ImagenetModelConfig,
    "vgg16-imagenet": Vgg16ImagenetModelConfig,
    "vgg16bn-imagenet": Vgg16bnImagenetModelConfig,
}
all_model_names = list(model_config_class_dict.keys())
