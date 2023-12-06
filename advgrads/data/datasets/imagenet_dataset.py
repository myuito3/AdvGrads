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

"""ImageNet dataset."""

from typing import List, Optional

from torchvision import transforms
from torchvision.datasets import ImageNet


DATA_PATH = "./data/imagenet"


class ImagenetDataset(ImageNet):
    """The ImageNet Dataset.

    Args:
        transform: Transform objects for image preprocessing.
        indices_to_use: List of image indices to be used.
    """

    def __init__(
        self,
        transform: transforms.Compose,
        indices_to_use: Optional[List[int]] = None,
    ) -> None:
        super().__init__(root=DATA_PATH, split="val", transform=transform)

        all_samples = self.samples
        self.samples = []
        for i in indices_to_use:
            self.samples.append(all_samples[i])

    @property
    def num_classes(self) -> int:
        return 1000
