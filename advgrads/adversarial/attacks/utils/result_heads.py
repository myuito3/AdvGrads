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

"""Result heads."""

from enum import Enum


class ResultHeadNames(Enum):
    """List of result outputs."""

    X_ADV = "x_adv"
    SHAPE = "shape"
    PREDS = "preds"
    SUCCEED = "succeed"
    NUM_SUCCEED = "num_succeed"
    SUCCESS_RATE = "success_rate"

    QUERIES = "queries"
    QUERIES_SUCCEED = "queries_succeed"
    MEAN_QUERY = "mean_query"
    MEDIAN_QUERY = "median_query"
