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

"""Init attack/defense method configs."""

from advgrads.adversarial.attacks.base_attack import AttackConfig
from advgrads.adversarial.attacks.deepfool import DeepFoolAttackConfig
from advgrads.adversarial.attacks.fgsm import FgsmAttackConfig
from advgrads.adversarial.attacks.i_fgsm import IFgsmAttackConfig
from advgrads.adversarial.attacks.mi_fgsm import MiFgsmAttackConfig
from advgrads.adversarial.attacks.ni_fgsm import NiFgsmAttackConfig
from advgrads.adversarial.attacks.pgd import PGDAttackConfig
from advgrads.adversarial.attacks.signhunter import SignHunterAttackConfig
from advgrads.adversarial.attacks.simba import SimBAAttackConfig
from advgrads.adversarial.attacks.square import SquareAttackConfig
from advgrads.adversarial.defenses.input_transform.base_defense import DefenseConfig
from advgrads.adversarial.defenses.input_transform.bit_depth_reduction import (
    BitDepthReductionDefenseConfig,
)
from advgrads.adversarial.defenses.input_transform.jpeg_compression import (
    JpegCompressionDefenseConfig,
)
from advgrads.adversarial.defenses.input_transform.randomization import (
    RandomizationDefenseConfig,
)


def get_attack_config_class(name: str) -> AttackConfig:
    assert name in all_attack_names, f"Attack method named '{name}' not found."
    return attack_class_dict[name]


def get_defense_config_class(name: str) -> DefenseConfig:
    assert name in all_defense_names, f"Defense method named '{name}' not found."
    return defense_class_dict[name]


attack_class_dict = {
    "deepfool": DeepFoolAttackConfig,
    "fgsm": FgsmAttackConfig,
    "i-fgsm": IFgsmAttackConfig,
    "mi-fgsm": MiFgsmAttackConfig,
    "ni-fgsm": NiFgsmAttackConfig,
    "pgd": PGDAttackConfig,
    "signhunter": SignHunterAttackConfig,
    "simba": SimBAAttackConfig,
    "square": SquareAttackConfig,
}
all_attack_names = list(attack_class_dict.keys())

defense_class_dict = {
    "bit-red": BitDepthReductionDefenseConfig,
    "jpeg": JpegCompressionDefenseConfig,
    "randomization": RandomizationDefenseConfig,
}
all_defense_names = list(defense_class_dict.keys())
