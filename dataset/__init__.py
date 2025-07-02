""" This module provides utility functions for working with datasets. """
__all__ = [
    'mmlu_pro',
    'mmlu',
    'MMLUDataset',
    'MMLUProDataset',
    'BaseDataset'
]

from . import mmlu_pro, mmlu
from .mmlu import MMLUDataset
from .mmlu_pro import MMLUProDataset
from .abc import BaseDataset


class AccMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter."""
        self.num_correct = 0
        self.num_total = 0

    def update(self, result: bool):
        """Updates the meter with a new result.

        Args:
            result (bool): Whether the result is correct (True) or incorrect (False).
        """
        self.num_correct += int(result)
        self.num_total += 1

    @property
    def acc(self):
        """Calculates and returns the accuracy."""
        return self.num_correct / self.num_total if self.num_total > 0 else 0.0
