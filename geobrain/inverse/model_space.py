"""``JointModelSpace``: one model mesh, N physics meshes, zero hand-wiring.

The cross-mesh joint-inversion facade of the L2 campaign: every physics
term in a joint inversion runs on ITS OWN optimal mesh (gravity coarse,
DC unstructured near-surface, EM padded, …) while the inversion parameters
live once on a MODEL mesh. This object owns exactly that wiring:

- one :class:`~geobrain.mesh.projection.MeshProjection` per physics
  term, built EAGERLY at construction so coverage problems surface
  immediately (the projection layer's construction-time warnings and the
  ``padding='raise'`` default do the loud-failure work);
- :meth:`adapter`: the differentiable ``Tensor -> Tensor`` callable that
  plugs straight into :class:`~geobrain.inverse.joint_binding.JointForward`'s
  ``field_to_mesh=`` seam;
- :meth:`forward`: the one-liner that wraps an operator into a
  ``JointForward`` term with the right adapter attached;
- :meth:`pullback`: the exact adjoint ``Wᵀ`` back to the model mesh
  (sensitivity/gradient transport for diagnostics and preconditioners);
- :meth:`coverage_report`: the per-term coverage summary.

Minimal usage (the whole cross-mesh wiring of a joint problem)::

    space = JointModelSpace(model_mesh, {
        "grav": grav_mesh,                       # defaults
        "dc":   {"mesh": dc_mesh, "method": "volume", "padding": "border"},
    })
    problem = JointProblem(
        model=model,
        forwards={
            "grav": space.forward("grav", grav_op),
            "dc":   space.forward("dc", dc_op, fields={"sig": "sigma"}),
        },
        observed=..., likelihoods=...,
    )

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

import torch

from ..core.errors import GeoBrainError
from ..mesh.base import Mesh
from ..mesh.projection import MeshProjection
from ..core.operator import Operator
from .joint_binding import JointForward

__all__ = ["JointModelSpace"]

_SPEC_KEYS = frozenset({
    "mesh", "method", "padding", "conservative", "k_neighbors", "idw_power",
})


class _MeshProjectionLike(Protocol):
    """Typed structural surface used from the skipped core implementation."""

    target: Mesh

    def project(self, field: torch.Tensor) -> torch.Tensor: ...

    def as_sparse_matrix(self) -> torch.Tensor: ...

    def coverage_report(self) -> dict[str, int | float | str]: ...


class JointModelSpace:
    """One model mesh + per-physics target meshes with cached projections."""

    def __init__(
        self,
        model_mesh: Mesh,
        physics: Mapping[str, "Mesh | Mapping[str, Any]"],
        *,
        conservative: bool = True,
        padding: str = "raise",
    ) -> None:
        """
        Args:
            model_mesh: the mesh the inversion parameters live on.
            physics: ``{name: mesh}`` or ``{name: {"mesh": mesh, ...}}``,
                per-term overrides among ``method`` / ``padding`` /
                ``conservative`` / ``k_neighbors`` / ``idw_power`` (all
                forwarded to :class:`MeshProjection`).
            conservative / padding: defaults for terms without overrides.
        """
        if not isinstance(model_mesh, Mesh):
            raise GeoBrainError(
                "model_mesh must be a Mesh",
                object_name="JointModelSpace", field="model_mesh",
                expected="Mesh", actual=type(model_mesh).__name__,
            )
        if not physics:
            raise GeoBrainError(
                "physics must name at least one term",
                object_name="JointModelSpace", field="physics",
                expected="non-empty mapping", actual=physics,
            )
        self.model_mesh = model_mesh
        self._projections: dict[str, _MeshProjectionLike] = {}
        for name, spec in physics.items():
            if isinstance(spec, Mesh):
                spec = {"mesh": spec}
            unknown = set(spec) - _SPEC_KEYS
            if unknown or "mesh" not in spec:
                raise GeoBrainError(
                    f"physics['{name}'] must be a Mesh or a dict with 'mesh' "
                    f"(+ optional {sorted(_SPEC_KEYS - {'mesh'})})",
                    object_name="JointModelSpace", field=f"physics['{name}']",
                    expected="Mesh | {'mesh': Mesh, ...}",
                    actual=sorted(unknown) or "missing 'mesh'",
                )
            kw = dict(spec)
            mesh = kw.pop("mesh")
            kw.setdefault("conservative", conservative)
            kw.setdefault("padding", padding)
            # Built EAGERLY: a coverage problem in ANY term should stop the
            # study at construction, not surface mid-inversion.
            self._projections[name] = MeshProjection(
                model_mesh, mesh, field_name="field", **kw,
            )

    # ------------------------------------------------------------- surface

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._projections)

    def projection(self, name: str) -> MeshProjection:
        """The term's cached ``model_mesh -> physics_mesh`` projection."""
        try:
            return self._projections[name]
        except KeyError:
            raise GeoBrainError(
                f"unknown physics term '{name}'",
                object_name="JointModelSpace", field="name",
                expected=f"one of {sorted(self._projections)}", actual=name,
            ) from None

    def mesh(self, name: str) -> Mesh:
        return self.projection(name).target

    def adapter(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Differentiable field adapter for ``JointForward(field_to_mesh=...)``.

        Handles single fields AND leading-dim batches (the projection layer's
        batched path), so the same term drops into deterministic inversion
        and particle-ensemble (SVGD) drivers unchanged.
        """
        proj: _MeshProjectionLike = self.projection(name)
        return proj.project

    def forward(
        self,
        name: str,
        op: Operator,
        *,
        fields: Mapping[str, str] | None = None,
        output: str | None = None,
        ctx_overrides: Mapping[str, Any] | None = None,
    ) -> JointForward:
        """Wrap ``op`` into a :class:`JointForward` term with this space's adapter."""
        return JointForward(
            op, fields=fields, output=output,
            field_to_mesh=self.adapter(name),
            ctx_overrides=ctx_overrides,
        )

    def pullback(self, name: str, field: torch.Tensor) -> torch.Tensor:
        """Exact adjoint transport ``Wᵀ x``: physics-mesh field → model mesh.

        The mathematical adjoint of the term's projection (identical to the
        forward's autograd backward), sensitivity/gradient transport for
        diagnostics, preconditioners, or hand-rolled updates. Returns the
        model-mesh grid layout when the model mesh carries one.
        """
        proj = self.projection(name)
        w = proj.as_sparse_matrix()
        out = torch.sparse.mm(
            w.t(), field.reshape(-1).to(torch.float64).unsqueeze(1)
        ).reshape(-1)
        try:
            # UnstructuredMesh.shape is a raising tombstone: flat output.
            # AttributeError is the tombstone contract; anything else from a
            # future shape property is a real bug and must propagate.
            shape = tuple(getattr(self.model_mesh, "shape"))
        except AttributeError:
            return out
        return out.reshape(shape)

    def coverage_report(
        self,
    ) -> dict[str, dict[str, int | float | str]]:
        """``{name: MeshProjection.coverage_report()}`` for every term."""
        return {
            name: proj.coverage_report()
            for name, proj in self._projections.items()
        }

    @staticmethod
    def cross_gradient(
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        weight: float = 1.0,
        normalized: bool = True,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:
        """Structural coupling of two MODEL-mesh fields (Gallardo–Meju).

        Convenience passthrough to
        :func:`geobrain.optim.regularizers.cross_gradient`, with a
        ``JointModelSpace`` the coupled fields are already co-located on the
        model mesh, so structural coupling needs no projection at all: add
        ``space.cross_gradient(state["a"], state["b"], weight=...)`` to the
        loss. (Petrophysical coupling, one shared state driving several
        derived fields through EarthModel Links; needs even less: nothing.)
        """
        from ..optim.regularizers import cross_gradient as _xg

        return _xg(
            a,
            b,
            weight=weight,
            normalized=normalized,
            eps=eps,
        )
