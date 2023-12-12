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

"""Eval metrics."""

import torch


class SuccessRateMeter:
    """Success rate meter."""

    def __init__(self) -> None:
        self.reset()

    def __str__(self) -> str:
        return "Success Rate: {}/{} ({:.2f}%) ".format(
            self.count_succeed, self.count_all, self.get_success_rate()
        )

    def reset(self) -> None:
        """Function to initialize metrics."""
        self.count_succeed = 0
        self.count_all = 0

    def update(self, count_succeed: int, count_all: int) -> None:
        """Function to update metrics.

        Args:
            count_succeed: Number of successful attacks.
            count_all: Number of experiments.
        """
        self.count_succeed += count_succeed
        self.count_all += count_all

    def get_success_rate(self) -> float:
        """Calculate the success rate."""
        return float(self.count_succeed / self.count_all * 100.0)


class QueryMeter:
    """Query meter."""

    def __init__(self) -> None:
        self.reset()

    def __str__(self) -> str:
        return (
            "Mean Query: {:.2f} Median Query: {:.2f} ".format(
                self.get_mean(), self.get_median()
            )
            if self.queries
            else ""
        )

    def reset(self) -> None:
        """Function to initialize metrics."""
        self.queries = []

    def update(self, queries: list) -> None:
        """Function to update metrics.

        Args:
            queries: A list of the number of queries that were attacked successfully.
        """
        self.queries.extend(queries)

    def get_mean(self) -> float:
        """Calculate the mean from the stacked queries."""
        return float(torch.mean(torch.tensor(self.queries).float()))

    def get_median(self) -> float:
        """Calculate the median from the stacked queries."""
        return float(torch.median(torch.tensor(self.queries).float()))
