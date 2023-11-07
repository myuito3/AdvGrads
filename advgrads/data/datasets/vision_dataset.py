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

"""Vision Dataset."""

from typing import List, Optional, Tuple

import numpy as np
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10


DATA_PATH = "./data/vision_datasets"


class MnistDataset(Dataset):
    """The MNIST Dataset."""

    def __init__(self, indices_to_use: Optional[List[int]] = None) -> None:
        super().__init__()
        data = MNIST(root=DATA_PATH, train=False, download=True)
        arrays = (data.data.numpy(), data.targets.numpy())
        if indices_to_use is not None:
            arrays = tuple([array[indices_to_use] for array in arrays])

        self.data = arrays[0]
        self.targets = arrays[1]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return F.to_tensor(self.data[index]), self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_classes(self) -> int:
        return 10


class Cifar10Dataset(Dataset):
    """The CIFAR-10 Dataset."""

    def __init__(self, indices_to_use: list = None) -> None:
        super().__init__()
        data = CIFAR10(root=DATA_PATH, train=False, download=True)
        arrays = (data.data, np.array(data.targets, dtype=np.longlong))
        if indices_to_use is not None:
            arrays = tuple([array[indices_to_use] for array in arrays])

        self.data = arrays[0]
        self.targets = arrays[1]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return F.to_tensor(self.data[index]), self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_classes(self) -> int:
        return 10
