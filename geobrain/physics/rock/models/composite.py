"""
End-to-end rock-physics workflow.

Uses our function-style
registry (:func:`~geobrain.physics.rock.models.create_model`) in place of
the ``ModelRegistry`` class.

:class:`RockPhysicsWorkflow` orchestrates the standard pipeline:

    (φ, Sw, P) → mineral mixing → dry rock → fluid mixing →
                 fluid substitution → (Vp, Vs, ρ_bulk)

Each stage is configurable via :class:`RockPhysicsConfig`. Workflow
dry-rock models must expose a ``compute_dry_rock(K0, G0, phi, **params)``
method, T2a SoftSand/StiffSand/HertzMindlin/DEM/CriticalPorosity and
the T6 contact-cement family all do.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RockPhysicsConfig
from ._base import CompositeModel
from ._registry import create_model
from ._types import BRIE_E, CN, P as P_DEFAULT, PHI_C, as_tensor
from ..data.materials import get_fluid, get_mineral
from ..moduli import v_from_moduli, validate_porosity


class RockPhysicsWorkflow(CompositeModel):
    """
    End-to-end rock-physics pipeline.

    Args:
        config: :class:`RockPhysicsConfig` or plain dict.

    Example::

        wf = RockPhysicsWorkflow.from_preset("shaly_sand")
        Vp, Vs, rho = wf(phi=0.2, Sw=0.8, P=30.0)
    """

    def __init__(self, config: RockPhysicsConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            from ..config import RockPhysicsConfig

            config = RockPhysicsConfig.from_dict(config)
        self.config = config
        self._setup_components()

    def _setup_components(self) -> None:
        for label, default in (
            ("dry_rock", "SoftSand"),
            ("fluid_sub", "Gassmann"),
            ("fluid_mix", "Wood"),
        ):
            model_name = self.config.models.get(label, default)
            instance = create_model(model_name, **self._model_params(model_name))
            setattr(self, f"_{label}", instance)

    def _model_params(self, model_name: str) -> dict[str, Any]:
        """Map workflow config to model-constructor kwargs."""
        params: dict[str, Any] = {}
        if model_name == "CriticalPorosity":
            params["phi_c"] = self.config.parameters.get("phi_c", PHI_C)
        elif model_name == "DEM":
            if "n_steps" in self.config.parameters:
                params["n_steps"] = self.config.parameters["n_steps"]
        elif model_name in ("ContactCement", "ContactCementFull", "MUHS", "PCM"):
            if "scheme" in self.config.parameters:
                params["scheme"] = self.config.parameters["scheme"]
        elif model_name == "Brie":
            params["exponent"] = self.config.parameters.get("e", BRIE_E)
        return params

    # --- pipeline stages ----------------------------------------------------

    def _compute_minerals(self) -> tuple[Tensor, Tensor, Tensor]:
        """Effective mineral moduli + density via the configured mixer."""
        mixer = self.config.models.get("mineral_mixer", "VRH")
        names = list(self.config.minerals.keys())
        fractions = list(self.config.minerals.values())

        K_list, G_list, rho_list = [], [], []
        for name in names:
            m = get_mineral(name)
            K_list.append(m.K)
            G_list.append(m.G)
            rho_list.append(m.rho)
        K = as_tensor(K_list)
        G = as_tensor(G_list)
        rho = as_tensor(rho_list)
        f = as_tensor(fractions)

        rho0 = torch.sum(f * rho)
        if mixer == "VRH":
            K0 = 0.5 * (torch.sum(f * K) + 1.0 / torch.sum(f / (K + 1e-10)))
            G0 = 0.5 * (torch.sum(f * G) + 1.0 / torch.sum(f / (G + 1e-10)))
        elif mixer == "Voigt":
            K0 = torch.sum(f * K)
            G0 = torch.sum(f * G)
        elif mixer == "Reuss":
            K0 = 1.0 / torch.sum(f / (K + 1e-10))
            G0 = 1.0 / torch.sum(f / (G + 1e-10))
        else:
            raise ValueError(
                f"Unknown mineral mixer: {mixer!r}. Use 'VRH', 'Voigt', or 'Reuss'."
            )
        return K0, G0, rho0

    def _compute_fluid(self, Sw: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Effective fluid ``(K, ρ)`` for the configured composition.

        Conventions:
            - ``Sw`` is always the water-phase saturation by name.
            - 1-fluid configs return that fluid's properties directly.
            - 2-fluid configs use the configured ``fluid_mix`` model
              (Wood / Brie). Water is the phase named ``"water"`` when
              present, otherwise the first listed fluid.
            - ≥3-fluid configs use Reuss (Wood) on the config fractions,
              with the water fraction replaced by ``Sw`` if given.
        """
        names = list(self.config.fluids.keys())

        if len(names) == 1:
            fl = get_fluid(names[0])
            return as_tensor(fl.K), as_tensor(fl.rho)

        if len(names) == 2:
            water_name = "water" if "water" in self.config.fluids else names[0]
            other_name = next(n for n in names if n != water_name)
            fl_w = get_fluid(water_name)
            fl_o = get_fluid(other_name)
            if Sw is None:
                Sw = as_tensor(self.config.fluids.get(water_name, 0.5))
            return self._fluid_mix.compute_fluid_mix(
                fl_w.K, fl_o.K, fl_w.rho, fl_o.rho, Sw,
            )

        # N >= 3 fluids: Wood (Reuss) mixing on config fractions.
        if Sw is not None and torch.is_tensor(Sw) and Sw.ndim > 0:
            raise ValueError(
                f"RockPhysicsWorkflow._compute_fluid: a vector Sw "
                f"({tuple(Sw.shape)}) is only supported for 2-phase fluid "
                f"configurations; got {len(names)} fluids ({names}). "
                f"Use a scalar Sw, restrict to two phases, or pass per-phase "
                f"saturations explicitly."
            )
        K_vals, rho_vals, f_vals = [], [], []
        for name in names:
            fl = get_fluid(name)
            K_vals.append(fl.K)
            rho_vals.append(fl.rho)
            f_vals.append(self.config.fluids[name])
        K = as_tensor(K_vals)
        rho = as_tensor(rho_vals)
        f = as_tensor(f_vals)

        if Sw is not None:
            Sw = as_tensor(Sw)
            water_idx = (
                names.index("water") if "water" in self.config.fluids else 0
            )
            remaining = 1.0 - Sw
            other_mask = torch.ones(len(names), dtype=torch.bool)
            other_mask[water_idx] = False
            f_rest_sum = f[other_mask].sum()
            new_f = torch.zeros(len(names), dtype=Sw.dtype, device=Sw.device)
            new_f[water_idx] = Sw
            if float(f_rest_sum) > 0:
                new_f[other_mask] = f[other_mask] / f_rest_sum * remaining
            f = new_f

        K_eff = 1.0 / torch.sum(f / (K + 1e-10))
        rho_eff = torch.sum(f * rho)
        return K_eff, rho_eff

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        phi: float | Tensor,
        Sw: float | Tensor | None = None,
        P: float | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return ``(Vp, Vs, ρ_bulk)`` for a given porosity (+ optional Sw, P).

        ``Vp / Vs`` are in m/s; ``ρ_bulk`` is in SI kg/m³; this workflow
        passes the (SI) ``get_mineral``/``get_fluid`` density values straight
        through with no internal conversion.
        """
        phi = validate_porosity(as_tensor(phi), clamp=True)
        if Sw is not None:
            Sw = validate_porosity(as_tensor(Sw), name="Sw", clamp=True)

        phi_c = self.config.parameters.get("phi_c", PHI_C)
        Cn = self.config.parameters.get("Cn", CN)
        if P is None:
            P = self.config.parameters.get("P", P_DEFAULT)

        K0, G0, rho0 = self._compute_minerals()
        K_dry, G_dry = self._dry_rock.compute_dry_rock(
            K0, G0, phi, phi_c=phi_c, Cn=Cn, P=P, **kwargs,
        )
        K_fl, rho_fl = self._compute_fluid(Sw)

        # Gassmann fluid substitution (stiffness form) via the single shared
        # core: the registered ``Gassmann`` ComponentModel uses the T2a
        # operator-friendly signature (K_dry, phi, K_fluid, rho_fluid → vp, rho);
        # here we need K_sat directly to feed v_from_moduli below.
        from ._fluid_gassmann import gassmann_k_sat
        K_sat = gassmann_k_sat(K_dry, K0, K_fl, phi)
        G_sat = G_dry  # Gassmann invariance

        rho_bulk = (1 - phi) * rho0 + phi * rho_fl
        Vp, Vs = v_from_moduli(K_sat, G_sat, rho_bulk)
        return Vp, Vs, rho_bulk

    @classmethod
    def from_preset(cls, name: str, **overrides) -> RockPhysicsWorkflow:
        """Build a workflow from a preset, optionally overriding parameters."""
        from ..config import get_preset

        config = get_preset(name)
        if overrides:
            config = config.with_overrides(**overrides)
        return cls(config)

    @staticmethod
    def list_presets() -> list[str]:
        from ..config import PRESETS

        return sorted(PRESETS.keys())

    def __repr__(self) -> str:
        minerals = ", ".join(f"{k}:{v:.0%}" for k, v in self.config.minerals.items())
        return (
            f"RockPhysicsWorkflow(minerals=[{minerals}], "
            f"model={self.config.models.get('dry_rock', 'SoftSand')})"
        )


__all__ = ["RockPhysicsWorkflow"]
