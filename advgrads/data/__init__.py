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

"""Init datasets."""

from torch.utils.data import Dataset

from advgrads.data.datasets.imagenet_dataset import ImagenetDataset
from advgrads.data.datasets.vision_dataset import (
    MnistDataset,
    Cifar10Dataset,
)


def get_dataset_class(name: str) -> Dataset:
    assert name in all_dataset_names, f"Dataset named '{name}' not found."
    return dataset_class_dict[name]


dataset_class_dict = {
    "cifar10": Cifar10Dataset,
    "imagenet": ImagenetDataset,
    "mnist": MnistDataset,
}
all_dataset_names = list(dataset_class_dict.keys())
