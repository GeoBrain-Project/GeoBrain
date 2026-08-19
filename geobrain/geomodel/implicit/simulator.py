"""
Lightweight Simulator wrapper around ImplicitModel.

Rather than registering the simulator with a global
``Simulator.create('implicit', ...)`` factory, this package uses direct
class imports throughout, so the registry hook has been removed. The class
is now a plain Python object exposing a simple ``simulate(config)`` API.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import torch
from typing import Callable, Dict, List, Optional, cast

from .data import ImplicitModelConfig, SeriesDefinition, FaultDefinition
from .model import ImplicitModel


class ImplicitSimulator:
    """
    Simulator wrapper for :class:`ImplicitModel`.

    Provides a thin ``simulate(config)`` entry point so callers can pass an
    :class:`ImplicitModelConfig` directly and get the block model back.

    Args:
        series: List of SeriesDefinition objects (geological layers).
        faults: Optional list of FaultDefinition objects.
        soft: Use differentiable soft classification (default True).
        temperature: Sigmoid sharpness for soft classification.
        transform: Optional output transform applied to the block tensor.

    Example:
        >>> sim = ImplicitSimulator(series=[series_def])
        >>> result = sim.simulate(config)
        >>> block = result['block']
    """

    def __init__(
        self,
        series: List[SeriesDefinition],
        faults: Optional[List[FaultDefinition]] = None,
        soft: bool = True,
        temperature: float = 50.0,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.series = series
        self.faults = faults
        self.soft = soft
        self.temperature = temperature
        self.transform = transform
        self._model: Optional[ImplicitModel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        config: Optional[ImplicitModelConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run an implicit geological model simulation.

        Args:
            config: ImplicitModelConfig (uses default if None).

        Returns:
            Dict with ``'block'``, ``'scalar_fields'``, and ``'grid'`` tensors.

        Note:
            When a ``transform`` is configured it applies only to the
            tensor-valued outputs (``'block'``, ``'grid'``) and is skipped
            for list-valued outputs (``'scalar_fields'``, ``'iso_vals'``).
            This matches the ``Callable[[Tensor], Tensor]`` transform
            contract.
        """
        if config is None:
            config = ImplicitModelConfig()

        self._model = ImplicitModel(
            config=config,
            series=self.series,
            faults=self.faults,
        )
        result = self._model(soft=self.soft, temperature=self.temperature)

        if self.transform is not None:
            if isinstance(result, dict):
                result = {
                    k: self.transform(v) if isinstance(v, torch.Tensor) else v
                    for k, v in result.items()
                }
            else:
                result = self.transform(result)
        return cast(Dict[str, torch.Tensor], result)

    def __call__(
        self,
        config: Optional[ImplicitModelConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.simulate(config)

    # ------------------------------------------------------------------
    # Capability flags (kept for parity with other geomodel simulators)
    # ------------------------------------------------------------------

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def supports_conditioning(self) -> bool:
        return True

    @property
    def model(self) -> Optional[ImplicitModel]:
        """Access the underlying ImplicitModel after the first simulate()."""
        return self._model
