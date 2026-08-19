"""Shared closed Agent contract for classical simulation façades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np

from ...capabilities import GeomodelCapabilityReport
from ...frames import GeoFrame, Geometry
from ...errors import GeomodelContractError
from ...resources import GeomodelResourceEstimate


class SimulationAgentContract:
    """Report/schema/resource methods inherited by production simulators."""

    @classmethod
    def capabilities(cls) -> GeomodelCapabilityReport:
        return GeomodelCapabilityReport(
            cls.__name__, "geomodel", "production", cls.__name__,
            (2, 3), ("continuous", "categorical"), ("1", None), "m",
            ("float64",), ("cpu",), ("indexed", "exhaustive", "dense", "fft", "training_image"),
            ("hard", "soft"),
            "fixed seed, execution backend, and declared numerical policies",
            "not differentiable", (), False, True, (),
        )

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "geobrain.geomodel.input/1.0",
            "type": "object",
            "properties": {
                "data": {"type": ["object", "null"]},
                "domain": {"type": "object"},
            },
            "required": ["domain"],
            "additionalProperties": False,
        }

    def estimate_resources(self, domain: object) -> GeomodelResourceEstimate:
        geometry = domain.geometry if isinstance(domain, GeoFrame) else domain
        if isinstance(geometry, Geometry):
            nodes = int(geometry.npoints)
        else:
            values = np.asarray(domain)
            if values.ndim != 2:
                raise GeomodelContractError(
                    "simulation resource estimate requires a geometry or coordinate matrix",
                    object_name=type(self).__name__, field="domain",
                    expected="Geometry, GeoFrame, or rank-2 coordinates", actual=tuple(values.shape),
                )
            nodes = int(values.shape[0])
        workers = int(getattr(getattr(self, "execution", None), "workers", 1))
        parts = (("coordinates_and_output", nodes * 40 * workers),)
        return GeomodelResourceEstimate(
            nodes * 40 * workers, parts, 0, 0, 0, None, 0, 0, workers,
            ("family-generic lower-bound estimate",),
        )


__all__ = ["SimulationAgentContract"]
