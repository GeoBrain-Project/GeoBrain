"""
Configuration + lithology presets for :class:`RockPhysicsWorkflow`.

Three public symbols:

- :class:`RockPhysicsConfig`: dataclass holding minerals, fluids, model
  choices, and numeric parameters.
- :data:`PRESETS`: dict of 10 lithology configs (``clean_sand``,
  ``shaly_sand``, ``gas_sand``, ``tight_sand``, ``limestone``, ``dolomite``,
  ``shale``, ``gas_shale``, ``chalk``, ``cemented_sand``).
- :func:`get_preset`, :func:`list_presets`: case-insensitive lookup
  utilities; ``get_preset`` returns a deep copy so callers can mutate
  without affecting the global table.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from .models._types import BRIE_E, CN, F, P, PHI_C


@dataclass
class RockPhysicsConfig:
    """
    Configuration for a rock-physics workflow.

    Attributes:
        minerals: Volume fractions of minerals (auto-normalised to sum to 1).
        fluids:   Volume fractions of fluids (auto-normalised to sum to 1).
        models:   Names of the four pipeline stages
                  ``{"mineral_mixer", "dry_rock", "fluid_sub", "fluid_mix"}``.
        parameters: Numeric knobs (``phi_c, Cn, P, f, e``, …).
    """

    minerals: dict[str, float]
    fluids: dict[str, float]
    models: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Own the mapping state: defaulting and normalisation mutate these
        # dicts, and mutating a caller-supplied dict in place would silently
        # rescale the caller's own data.
        self.minerals = dict(self.minerals)
        self.fluids = dict(self.fluids)
        self.models = dict(self.models)
        self.parameters = dict(self.parameters)
        self._set_defaults()
        self._normalize()

    def _set_defaults(self) -> None:
        model_defaults = {
            "mineral_mixer": "VRH",
            "dry_rock": "SoftSand",
            "fluid_sub": "Gassmann",
            "fluid_mix": "Wood",
        }
        for key, value in model_defaults.items():
            self.models.setdefault(key, value)
        param_defaults = {
            "phi_c": PHI_C,
            "Cn": CN,
            "P": P,
            "f": F,
            "e": BRIE_E,
        }
        for key, value in param_defaults.items():
            self.parameters.setdefault(key, value)

    def _normalize(self) -> None:
        """
        Validate + normalise mineral / fluid fractions to sum to 1.

        Rejects negative fractions, empty composition, and zero/negative
        totals at construction so downstream code never sees nonphysical
        effective moduli.
        """
        for label, d in (("minerals", self.minerals), ("fluids", self.fluids)):
            if not d:
                raise ValueError(f"RockPhysicsConfig.{label} must not be empty.")
            for key, val in d.items():
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"RockPhysicsConfig.{label}[{key!r}] must be a number; "
                        f"got {type(val).__name__}."
                    )
                if val < 0:
                    raise ValueError(
                        f"RockPhysicsConfig.{label}[{key!r}] must be >= 0; got {val!r}."
                    )
            total = sum(d.values())
            if total <= 0:
                raise ValueError(
                    f"RockPhysicsConfig.{label} fractions must sum to a positive "
                    f"value; got total={total!r}."
                )
            for key in d:
                d[key] /= total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RockPhysicsConfig:
        return cls(
            minerals=d.get("minerals", {}),
            fluids=d.get("fluids", {}),
            models=d.get("models", {}),
            parameters=d.get("parameters", {}),
        )

    def copy(self) -> RockPhysicsConfig:
        return RockPhysicsConfig(
            minerals=copy.deepcopy(self.minerals),
            fluids=copy.deepcopy(self.fluids),
            models=copy.deepcopy(self.models),
            parameters=copy.deepcopy(self.parameters),
        )

    def with_overrides(self, **kwargs) -> RockPhysicsConfig:
        """
        Return a copy with overrides merged in.

        Recognised top-level keys: ``minerals`` / ``fluids`` (replace),
        ``models`` / ``parameters`` (dict-update). Anything else flows
        into ``parameters``.
        """
        new = self.copy()
        for key, value in kwargs.items():
            if key == "minerals":
                new.minerals = copy.deepcopy(value)
            elif key == "fluids":
                new.fluids = copy.deepcopy(value)
            elif key == "models":
                new.models.update(value)
            elif key == "parameters":
                new.parameters.update(value)
            else:
                new.parameters[key] = value
        new._normalize()
        return new

    def __repr__(self) -> str:
        minerals_str = ", ".join(f"{k}:{v:.0%}" for k, v in self.minerals.items())
        fluids_str = ", ".join(f"{k}:{v:.0%}" for k, v in self.fluids.items())
        return (
            f"RockPhysicsConfig(\n"
            f"  minerals=[{minerals_str}],\n"
            f"  fluids=[{fluids_str}],\n"
            f"  dry_rock={self.models.get('dry_rock')},\n"
            f"  phi_c={self.parameters.get('phi_c')}\n"
            f")"
        )


PRESETS: dict[str, RockPhysicsConfig] = {
    "clean_sand": RockPhysicsConfig(
        minerals={"quartz": 1.0}, fluids={"water": 1.0},
        models={"dry_rock": "SoftSand"}, parameters={"phi_c": 0.40},
    ),
    "shaly_sand": RockPhysicsConfig(
        minerals={"quartz": 0.7, "clay": 0.3}, fluids={"water": 1.0},
        models={"dry_rock": "SoftSand"}, parameters={"phi_c": 0.35},
    ),
    "gas_sand": RockPhysicsConfig(
        minerals={"quartz": 0.8, "clay": 0.2}, fluids={"gas": 0.7, "water": 0.3},
        models={"dry_rock": "SoftSand", "fluid_mix": "Brie"},
        parameters={"phi_c": 0.36, "e": 3.0},
    ),
    "tight_sand": RockPhysicsConfig(
        minerals={"quartz": 0.85, "clay": 0.15}, fluids={"gas": 0.8, "water": 0.2},
        models={"dry_rock": "StiffSand", "fluid_mix": "Brie"},
        parameters={"phi_c": 0.30},
    ),
    "limestone": RockPhysicsConfig(
        minerals={"calcite": 0.95, "clay": 0.05}, fluids={"water": 1.0},
        models={"dry_rock": "DEM"}, parameters={"phi_c": 0.35},
    ),
    "dolomite": RockPhysicsConfig(
        minerals={"dolomite": 0.9, "calcite": 0.1}, fluids={"water": 1.0},
        models={"dry_rock": "DEM"}, parameters={"phi_c": 0.35},
    ),
    "shale": RockPhysicsConfig(
        minerals={"clay": 0.6, "quartz": 0.3, "calcite": 0.1}, fluids={"water": 1.0},
        models={"dry_rock": "SoftSand"}, parameters={"phi_c": 0.30},
    ),
    "gas_shale": RockPhysicsConfig(
        minerals={"clay": 0.5, "quartz": 0.35, "kerogen": 0.15},
        fluids={"gas": 0.9, "water": 0.1},
        models={"dry_rock": "SoftSand", "fluid_mix": "Brie"},
        parameters={"phi_c": 0.25},
    ),
    "chalk": RockPhysicsConfig(
        minerals={"calcite": 1.0}, fluids={"water": 1.0},
        models={"dry_rock": "CriticalPorosity"}, parameters={"phi_c": 0.65},
    ),
    "cemented_sand": RockPhysicsConfig(
        minerals={"quartz": 0.9, "clay": 0.1}, fluids={"water": 1.0},
        models={"dry_rock": "ContactCement"},
        parameters={"phi_c": 0.36, "Cn": 9.0},
    ),
}


def get_preset(name: str) -> RockPhysicsConfig:
    """Return a deep copy of the named preset (case-insensitive)."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset: '{name}'. Available presets: {available}")
    return PRESETS[key].copy()


def list_presets() -> list[str]:
    """Sorted list of all preset names."""
    return sorted(PRESETS.keys())


__all__ = ["PRESETS", "RockPhysicsConfig", "get_preset", "list_presets"]
