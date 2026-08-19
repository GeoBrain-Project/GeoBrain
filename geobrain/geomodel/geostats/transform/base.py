"""
Transform base class + Pipeline composition.

A :class:`Transform`
reads from a :class:`GeoFrame`, fits internal parameters, and
returns a new GeoFrame with the same geometry plus (or modified)
property columns. Composition with the ``|`` operator yields a
:class:`Pipeline` whose ``fit``/``transform``/``inverse_transform``
chain through the steps in order.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...frames import GeoFrame


class Transform(ABC):
    """
    Abstract base class for column-wise GeoFrame transforms.

    Subclasses implement :meth:`fit` (learn parameters in place,
    return ``self``) and :meth:`transform` (apply, return a new
    GeoFrame). :meth:`inverse_transform` is optional.
    """

    @abstractmethod
    def fit(self, data: "GeoFrame") -> "Transform":
        """Learn parameters from ``data``; must return ``self``."""

    @abstractmethod
    def transform(self, data: "GeoFrame") -> "GeoFrame":
        """Apply the (already-fitted) transform; returns a new GeoFrame."""

    def fit_transform(self, data: "GeoFrame") -> "GeoFrame":
        """Fit on ``data`` then apply."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: "GeoFrame") -> "GeoFrame":
        """Reverse the transform: opt-in. Default: raise."""
        raise NotImplementedError(f"{type(self).__name__} is not invertible")

    def __or__(self, other: "Transform") -> "Pipeline":
        """``t1 | t2`` → :class:`Pipeline` of the two steps."""
        steps: list[Transform] = []
        for t in (self, other):
            if isinstance(t, Pipeline):
                steps.extend(t.steps)
            else:
                steps.append(t)
        return Pipeline(steps)


class Pipeline(Transform):
    """Sequential pipeline; created by composing transforms with ``|``."""

    def __init__(self, steps: list[Transform]) -> None:
        self.steps: list[Transform] = list(steps)
        self._fitted_steps: list[Transform] = []

    def fit(self, data: "GeoFrame") -> "Pipeline":
        """Fit each step sequentially, threading the transformed table forward."""
        self._fitted_steps = []
        current = data
        for step in self.steps:
            step.fit(current)
            current = step.transform(current)
            self._fitted_steps.append(step)
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        """Apply all fitted steps to ``data``."""
        current = data
        for step in self._fitted_steps:
            current = step.transform(current)
        return current

    def inverse_transform(self, data: "GeoFrame") -> "GeoFrame":
        """Apply inverse of every step in reverse order."""
        current = data
        for step in reversed(self._fitted_steps):
            current = step.inverse_transform(current)
        return current

    def __repr__(self) -> str:
        return "Pipeline(" + " | ".join(repr(s) for s in self.steps) + ")"


__all__ = ["Pipeline", "Transform"]
