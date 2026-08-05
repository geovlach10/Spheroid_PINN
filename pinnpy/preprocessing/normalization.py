"""Feature/target normalization for FCNN-based fitting.

Z-score normalization: rescales each column to zero mean and unit
variance using statistics computed once from a reference dataset, and
can invert predictions back to physical units.

Usage:
    normalizer = ZScoreNormalizer().fit(x_train)
    x_train_norm = normalizer.transform(x_train)
    ...
    prediction = normalizer.inverse_transform(model_output_norm)

Always fit on training data only, then reuse the same fitted instance
to transform any new data (validation points, evaluation grids, etc.)
before it enters the model, and to inverse_transform model outputs
back to physical units.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class NormalizerInterface(ABC):
    """Contract for interchangeable normalization strategies."""

    @abstractmethod
    def fit(self, data: torch.Tensor) -> "NormalizerInterface":
        """Compute and store normalization statistics from `data`.

        Args:
            data: Reference tensor of shape (N, D). Statistics are
                computed per column.

        Returns:
            self, so this chains: `ZScoreNormalizer().fit(data)`.
        """
        ...

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Map `data` into normalized space.

        Args:
            data: Tensor of shape (N, D), same D as used in `fit`.

        Returns:
            Normalized tensor, same shape as `data`.
        """
        ...

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Map `data` from normalized space back to physical units.

        Args:
            data: Tensor in normalized space, shape (N, D).

        Returns:
            Tensor in physical units, same shape as `data`.
        """
        ...


class ZScoreNormalizer(NormalizerInterface):
    """Per-column z-score normalization.

        transform:          x* = (x - mean) / std
        inverse_transform:  x  = x* * std + mean

    Call fit() once on your training data; mean/std are then fixed and
    reused by every later transform()/inverse_transform() call.
    """

    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "ZScoreNormalizer":
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        assert (self.mean is not None) and (self.std is not None), "call fit() before transform()"
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        assert (self.mean is not None) and (self.std is not None), "call fit() before inverse_transform()"
        return data * self.std + self.mean