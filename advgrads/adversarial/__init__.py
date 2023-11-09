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

"""Init adversarial attacks/defenses methods."""

from advgrads.adversarial.attacks.base_attack import AttackConfig
from advgrads.adversarial.attacks.fgsm import FgsmAttackConfig
from advgrads.adversarial.attacks.i_fgsm import IFgsmAttackConfig
from advgrads.adversarial.attacks.mi_fgsm import MiFgsmAttackConfig
from advgrads.adversarial.attacks.signhunter import SignHunterAttackConfig
from advgrads.adversarial.attacks.square import SquareAttackConfig


def get_attack_config_class(name: str) -> AttackConfig:
    return attack_class_dict[name]


attack_class_dict = {
    "fgsm": FgsmAttackConfig,
    "i_fgsm": IFgsmAttackConfig,
    "mi_fgsm": MiFgsmAttackConfig,
    "signhunter": SignHunterAttackConfig,
    "square": SquareAttackConfig,
}
all_attack_names = list(attack_class_dict.keys())
