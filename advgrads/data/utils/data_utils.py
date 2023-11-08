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

"""Data utils."""

import os
from typing import List, Union

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


def get_dataloader(dataset: Dataset, batch_size: int = 128):
    """Returns a dataloader containing all images in the dataset.

    Args:
        dataset: Dataset to be load.
        batch_size: Number of images per batch.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def save_images(images: Tensor, filenames: Union[str, List[str]]) -> None:
    """Save images.

    Args:
        images: The images to save.
        filenames: The filenames to save to.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    assert len(images) == len(filenames)

    for image, filename in zip(images, filenames):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_image(image, filename)
