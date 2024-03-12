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

from typing import List, Optional, Type

from torch.utils.data import Dataset

from advgrads.data.datasets.imagenet_dataset import ImagenetDataset
from advgrads.data.datasets.vision_dataset import (
    MnistDataset,
    Cifar10Dataset,
)
from advgrads.data.utils import index_samplers


def get_dataset_class(name: str) -> Type[Dataset]:
    assert name in all_dataset_names, f"Dataset named '{name}' not found."
    return dataset_class_dict[name]


def get_dataset(
    name: str,
    image_indices: Optional[List[int]] = None,
    num_images: Optional[int] = None,
    **kwargs,
) -> Dataset:
    if image_indices is None:
        if name == "imagenet":
            image_indices = (
                index_samplers.get_random(num_images, population=50000)
                if num_images is not None
                else None
            )
        else:
            image_indices = (
                index_samplers.get_random(num_images)
                if num_images is not None
                else None
            )
    dataset = get_dataset_class(name)(indices_to_use=image_indices, **kwargs)
    return dataset, image_indices


dataset_class_dict = {
    "cifar10": Cifar10Dataset,
    "imagenet": ImagenetDataset,
    "mnist": MnistDataset,
}
all_dataset_names = list(dataset_class_dict.keys())
