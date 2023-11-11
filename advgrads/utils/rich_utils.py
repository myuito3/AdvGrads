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

"""Additional rich ui components."""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

CONSOLE = Console(width=120)


def console_print(msg: Any) -> None:
    """Print message via rich console.

    Args:
        msg: Message outputed in terminal.
    """
    CONSOLE.print(msg)


def console_log(msg: Any) -> None:
    """Log message via rich console.

    Args:
        msg: Message outputed in terminal.
    """
    CONSOLE.log(msg)
