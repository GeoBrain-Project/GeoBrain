"""Packed 2-D acoustic frequency-domain Helmholtz facade.

The constant-density system is

``A(vp) p = q``, ``A = -L_h - omega**2 / vp**2``.

GeoBrain uses the engineering phasor ``exp(+i*omega*t)``. Outgoing 2-D fields
therefore follow ``-(i/4) H_0^(2)(k r)``. Point-source amplitudes are integrated
2-D strengths and assembly divides them by cell area. Public receiver pressure
is native complex data with packed axes ``(trace, frequency, component)``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, ClassVar, cast

import torch

from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    ForwardOperator,
    ForwardOutput,
    ModelState,
)
from geobrain.mesh import require_field_matches_mesh
from geobrain.mesh.capabilities import UniformMesh
from geobrain.physics.wave.capabilities import (
    WaveCapabilityReport,
    WaveUnsupportedCombination,
)
from geobrain.physics.wave.errors import WaveContractError, WaveNumericsError

from .assembly import build_helmholtz_2d_coo, build_packed_helmholtz_2d_rhs
from .solve import solve_helmholtz_system


class _FrozenJSONList(list[object]):
    """A JSON-encoder-compatible list that rejects mutation."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("GeoBrain schema is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]


class _FrozenJSONDict(dict[str, object]):
    """A JSON-encoder-compatible dict that rejects mutation."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("GeoBrain schema is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable  # type: ignore[assignment]


def _freeze_json(value: object) -> object:
    """Recursively freeze one JSON-value tree without breaking ``json.dumps``."""
    if isinstance(value, Mapping):
        return _FrozenJSONDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return _FrozenJSONList([_freeze_json(item) for item in value])
    return value


def _position(value: object, *, owner: str) -> tuple[float, float]:
    """Validate and own one public ``(x, z)`` coordinate."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise WaveContractError(
            "Helmholtz position must be an (x, z) tuple",
            object_name=owner,
            field="position",
            expected="two finite metres",
            actual=value,
        )
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise WaveContractError(
            "Helmholtz position values must be numeric",
            object_name=owner,
            field="position",
            actual=value,
        ) from exc
    if any(not math.isfinite(item) for item in result):
        raise WaveContractError(
            "Helmholtz position values must be finite",
            object_name=owner,
            field="position",
            actual=value,
        )
    return result


def _shot_id(value: object, *, owner: str) -> int:
    """Validate one non-negative packed shot identifier."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WaveContractError(
            "shot_id must be a non-negative integer",
            object_name=owner,
            field="shot_id",
            expected=">= 0",
            actual=value,
        )
    return value


@dataclass(frozen=True, slots=True)
class Helmholtz2DSource:
    """One packed point source with integrated complex strength.

    Attributes:
        position: ``(x, z)`` source location [m].
        amplitude: complex source amplitude.
        shot_id: source identifier grouping receivers.
    """

    position: tuple[float, float]
    amplitude: complex = 1.0 + 0.0j
    shot_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _position(self.position, owner=type(self).__name__))
        try:
            amplitude = complex(self.amplitude)
        except (TypeError, ValueError, OverflowError) as exc:
            raise WaveContractError(
                "source amplitude must be complex-compatible",
                object_name=type(self).__name__,
                field="amplitude",
                actual=self.amplitude,
            ) from exc
        if not math.isfinite(amplitude.real) or not math.isfinite(amplitude.imag):
            raise WaveContractError(
                "source amplitude must be finite",
                object_name=type(self).__name__,
                field="amplitude",
                actual=self.amplitude,
            )
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "shot_id", _shot_id(self.shot_id, owner=type(self).__name__))


@dataclass(frozen=True, slots=True)
class Helmholtz2DReceiver:
    """One packed pressure trace associated with exactly one shot.

    Attributes:
        position: ``(x, z)`` receiver location [m].
        shot_id: which source this receiver listens to.
    """

    position: tuple[float, float]
    shot_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _position(self.position, owner=type(self).__name__))
        object.__setattr__(self, "shot_id", _shot_id(self.shot_id, owner=type(self).__name__))


@dataclass(frozen=True, slots=True)
class Helmholtz2DSurvey:
    """Immutable packed acquisition, frequencies, boundary, and solver settings.

    Every packed source/receiver ``position`` is ``(x, z)`` metres in the
    mesh frame, x on mesh axis-1, z depth (positive down) on axis-0,
    measured from the mesh origin.

    Attributes:
        sources / receivers: the frequency-domain acquisition tables.
        frequencies: modelling frequencies [Hz].
        n_pml: PML thickness [cells] (legacy alias of ``pml_thickness``).
        abc: absorbing-boundary kind.
        pml_thickness / pml_decay_factor / pml_target_reflection: PML
            profile controls.
        return_wavefield: keep the full complex field per frequency.
        solver: sparse solver id for the Helmholtz systems.
        residual_tolerance: acceptance tolerance on the solve residual.
    """

    sources: tuple[Helmholtz2DSource, ...] = ()
    receivers: tuple[Helmholtz2DReceiver, ...] = ()
    frequencies: tuple[float, ...] = ()
    n_pml: int = 20
    abc: str = "sommerfeld"
    pml_thickness: int = 0
    pml_decay_factor: float = 2.0
    pml_target_reflection: float = 1.0e-6
    return_wavefield: bool = False
    solver: str = "direct-splu"
    residual_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if not self.sources or not all(
            isinstance(item, Helmholtz2DSource) for item in self.sources
        ):
            raise WaveContractError(
                "Helmholtz survey requires packed sources",
                object_name=type(self).__name__,
                field="sources",
                expected="non-empty tuple of Helmholtz2DSource",
                actual=self.sources,
            )
        if not self.receivers or not all(
            isinstance(item, Helmholtz2DReceiver) for item in self.receivers
        ):
            raise WaveContractError(
                "Helmholtz survey requires packed receivers",
                object_name=type(self).__name__,
                field="receivers",
                expected="non-empty tuple of Helmholtz2DReceiver",
                actual=self.receivers,
            )
        try:
            frequencies = tuple(float(item) for item in self.frequencies)
        except (TypeError, ValueError, OverflowError) as exc:
            raise WaveContractError(
                "frequencies must be numeric Hz values",
                object_name=type(self).__name__, field="frequencies"
            ) from exc
        if (
            not frequencies
            or any(not math.isfinite(item) or item <= 0.0 for item in frequencies)
            or len(set(frequencies)) != len(frequencies)
        ):
            raise WaveContractError(
                "frequencies must be unique positive finite Hz values",
                object_name=type(self).__name__,
                field="frequencies",
                actual=self.frequencies,
            )
        source_shots = {item.shot_id for item in self.sources}
        receiver_shots = {item.shot_id for item in self.receivers}
        expected_shots = set(range(max(source_shots) + 1))
        if source_shots != expected_shots or receiver_shots != expected_shots:
            raise WaveContractError(
                "source and receiver shot domains must match contiguous 0..n_shot-1",
                object_name=type(self).__name__,
                field="shot_id",
                expected=sorted(expected_shots),
                actual={"source": sorted(source_shots), "receiver": sorted(receiver_shots)},
            )
        if isinstance(self.n_pml, bool) or not isinstance(self.n_pml, int) or self.n_pml < 0:
            raise WaveContractError(
                "n_pml must be >= 0", object_name=type(self).__name__, field="n_pml", actual=self.n_pml
            )
        if self.abc not in ("sommerfeld", "dirichlet"):
            raise WaveContractError(
                "abc must be 'sommerfeld' or 'dirichlet'",
                object_name=type(self).__name__, field="abc", actual=self.abc
            )
        if (
            isinstance(self.pml_thickness, bool)
            or not isinstance(self.pml_thickness, int)
            or self.pml_thickness < 0
        ):
            raise WaveContractError(
                "pml_thickness must be >= 0",
                object_name=type(self).__name__, field="pml_thickness", actual=self.pml_thickness
            )
        if not math.isfinite(float(self.pml_decay_factor)) or self.pml_decay_factor <= 0:
            raise WaveContractError(
                "pml_decay_factor must be > 0",
                object_name=type(self).__name__, field="pml_decay_factor", actual=self.pml_decay_factor
            )
        if not math.isfinite(float(self.pml_target_reflection)) or not 0.0 < self.pml_target_reflection < 1.0:
            raise WaveContractError(
                "pml_target_reflection must be in (0, 1)",
                object_name=type(self).__name__, field="pml_target_reflection", actual=self.pml_target_reflection
            )
        if type(self.return_wavefield) is not bool:
            raise WaveContractError(
                "return_wavefield must be boolean",
                object_name=type(self).__name__, field="return_wavefield", actual=self.return_wavefield
            )
        if self.solver != "direct-splu":
            raise WaveContractError(
                "only the production-validated direct-splu solver is supported",
                object_name=type(self).__name__, field="solver", actual=self.solver
            )
        if not math.isfinite(float(self.residual_tolerance)) or self.residual_tolerance <= 0.0:
            raise WaveContractError(
                "residual_tolerance must be positive and finite",
                object_name=type(self).__name__, field="residual_tolerance", actual=self.residual_tolerance
            )
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "receivers", tuple(self.receivers))
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "pml_decay_factor", float(self.pml_decay_factor))
        object.__setattr__(self, "pml_target_reflection", float(self.pml_target_reflection))
        object.__setattr__(self, "residual_tolerance", float(self.residual_tolerance))

    @property
    def n_shot(self) -> int:
        """Number of contiguous packed shots."""
        return max(source.shot_id for source in self.sources) + 1

    @property
    def n_trace(self) -> int:
        """Number of packed receiver rows."""
        return len(self.receivers)


class Helmholtz2D(ForwardOperator):  # type: ignore[misc]
    """CPU float64 packed Helmholtz facade with an implicit complex VJP.

    Device: CPU / float64 only (scipy sparse LU); move inputs to CPU first.
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("vp",),
        output_keys=("p",),
        input_units={"vp": "m/s"},
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (UniformMesh,)

    def __init__(self, survey: Helmholtz2DSurvey) -> None:
        super().__init__()
        if not isinstance(survey, Helmholtz2DSurvey):
            raise WaveContractError(
                "Helmholtz2D survey must be Helmholtz2DSurvey",
                object_name=type(self).__name__, field="survey", actual=type(survey).__name__
            )
        self.survey = survey

    @staticmethod
    def amplitude(pressure: torch.Tensor) -> torch.Tensor:
        """Return an explicit magnitude view without replacing primary data."""
        if not isinstance(pressure, torch.Tensor) or not torch.is_complex(pressure):
            raise WaveContractError(
                "amplitude expects a complex pressure tensor",
                object_name="Helmholtz2D", field="pressure"
            )
        return pressure.abs()

    @staticmethod
    def phase(pressure: torch.Tensor) -> torch.Tensor:
        """Return phase in radians for the declared engineering phasor."""
        if not isinstance(pressure, torch.Tensor) or not torch.is_complex(pressure):
            raise WaveContractError(
                "phase expects a complex pressure tensor",
                object_name="Helmholtz2D", field="pressure"
            )
        return torch.angle(pressure)

    @classmethod
    def capabilities(cls) -> WaveCapabilityReport:
        """Return deterministic immutable Helmholtz discovery data."""
        return WaveCapabilityReport(
            physics="acoustic-frequency-domain",
            equation="constant-density Helmholtz 2-D",
            dimension=2,
            maturity="production",
            required_model_fields=(("vp", "m/s"),),
            components=("pressure",),
            dtypes=("float64",),
            devices=("cpu",),
            backends=("direct-splu",),
            boundaries=("sommerfeld", "dirichlet", "pml"),
            memory_strategies=(),
            differentiable_model_fields=("vp",),
            differentiable_wavelets=False,
            mesh_capabilities=("UniformMesh",),
            resource_estimate_supported=False,
            unsupported=(
                WaveUnsupportedCombination(
                    selection=(("model.device", "cuda"),),
                    reason="the validated direct SuperLU backend is CPU-only",
                    remediation="move vp to CPU before constructing ModelState",
                ),
                WaveUnsupportedCombination(
                    selection=(("model.dtype", "float32"),),
                    reason="the validated sparse solve is float64/complex128 only",
                    remediation="provide float64 vp explicitly",
                ),
                WaveUnsupportedCombination(
                    selection=(("solver", "iterative"),),
                    reason="no iterative convergence/gradient row is production-validated",
                    remediation="select direct-splu",
                ),
            ),
        )

    @classmethod
    def input_schema(cls) -> Mapping[str, object]:
        """Return an immutable JSON-safe Agent/UI input and result schema."""
        schema = {
            "title": "GeoBrain Helmholtz2D",
            "version": "0.2.0",
            "maturity": "production",
            "model": {
                "vp": {"unit": "m/s", "dtype": "float64", "axes": ["z", "x"], "exclusiveMinimum": 0.0}
            },
            "survey": {
                "source_positions": {"unit": "m", "axes": ["source", "coordinate"], "coordinate_order": ["x", "z"]},
                "source_shot_index": {"axes": ["source"], "dtype": "int64"},
                "source_amplitude": {"unit": "Pa", "dtype": "complex128", "axes": ["source"]},
                "receiver_positions": {"unit": "m", "axes": ["trace", "coordinate"], "coordinate_order": ["x", "z"]},
                "receiver_shot_index": {"axes": ["trace"], "dtype": "int64"},
                "frequencies_hz": {"unit": "Hz", "axes": ["frequency"], "exclusiveMinimum": 0.0},
            },
            "boundary": {"kind": ["sommerfeld", "dirichlet", "pml"]},
            "runtime": {"device": ["cpu"], "solver": ["direct-splu"]},
            "output": {
                "p": {"unit": "Pa", "dtype": "complex128", "complex": True, "axes": ["trace", "frequency", "component"], "components": ["pressure"]}
            },
            "phasor_convention": "exp(+i*omega*t)",
            "gradients": {"vp": "implicit-hermitian-adjoint"},
            "unsupported": cls.capabilities().to_dict()["unsupported"],
        }
        return cast(Mapping[str, object], _freeze_json(schema))

    @staticmethod
    def _grid_index(
        mesh: Any, x: float, z: float, who: str, index: int
    ) -> tuple[int, int]:
        """Map public metres to platform ``(iz, ix)`` cell indices."""
        nz, nx = cast(tuple[int, int], mesh.shape)
        origin_z, origin_x = cast(tuple[float, float], mesh.origin)
        iz = int(math.floor((z - origin_z) / float(mesh.dz)))
        ix = int(math.floor((x - origin_x) / float(mesh.dx)))
        if not 0 <= iz < nz or not 0 <= ix < nx:
            raise WaveContractError(
                f"Helmholtz2D {who} #{index} is outside the mesh",
                object_name="Helmholtz2D",
                field=f"{who}_position",
                expected=f"indices within {(nz, nx)}",
                actual=(iz, ix),
            )
        return iz, ix

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (vp,) = state.fetch("vp")
        mesh = ctx.require_mesh()
        if len(mesh.shape) != 2:
            raise WaveContractError(
                "Helmholtz2D requires a 2-D mesh",
                object_name=type(self).__name__, field="mesh", actual=mesh.shape
            )
        require_field_matches_mesh(mesh, vp, name="vp", owner=type(self).__name__)
        if vp.device.type != "cpu":
            raise WaveNumericsError(
                "Helmholtz2D direct-splu is CPU-only",
                object_name=type(self).__name__, field="vp.device",
                expected="cpu", actual=str(vp.device),
                hint="move vp to CPU before execution",
            )
        if vp.dtype is not torch.float64:
            raise WaveNumericsError(
                "Helmholtz2D requires explicit float64 vp",
                object_name=type(self).__name__, field="vp.dtype",
                expected="torch.float64", actual=str(vp.dtype),
                hint="convert vp to float64 explicitly; the solver never silently casts",
            )
        if not bool(torch.isfinite(vp.detach()).all()) or bool((vp.detach() <= 0.0).any()):
            raise WaveNumericsError(
                "Helmholtz2D vp must be positive and finite",
                object_name=type(self).__name__, field="vp",
                hint="repair invalid velocity cells before solving",
            )
        nz, nx = cast(tuple[int, int], mesh.shape)
        if self.survey.pml_thickness > 0 and 2 * self.survey.pml_thickness >= min(nz, nx):
            raise WaveContractError(
                f"pml_thickness={self.survey.pml_thickness} too large for mesh shape {mesh.shape}",
                object_name=type(self).__name__, field="pml_thickness",
                expected=f"< {min(nz, nx) / 2:g}", actual=self.survey.pml_thickness,
            )
        source_indices = tuple(
            self._grid_index(mesh, source.position[0], source.position[1], "source", index)
            for index, source in enumerate(self.survey.sources)
        )
        receiver_indices = tuple(
            self._grid_index(mesh, receiver.position[0], receiver.position[1], "receiver", index)
            for index, receiver in enumerate(self.survey.receivers)
        )
        amplitudes = torch.tensor(
            [source.amplitude for source in self.survey.sources], dtype=torch.complex128
        )
        rhs = build_packed_helmholtz_2d_rhs(
            sources_iz_ix=source_indices,
            source_shot_index=tuple(source.shot_id for source in self.survey.sources),
            amplitudes=amplitudes,
            n_shot=self.survey.n_shot,
            nz=nz,
            nx=nx,
            cell_area=float(mesh.dz) * float(mesh.dx),
        )
        frequency_records: list[torch.Tensor] = []
        wavefields: list[torch.Tensor] = []
        use_pml = self.survey.pml_thickness > 0
        for frequency_hz in self.survey.frequencies:
            omega = 2.0 * math.pi * frequency_hz
            matrix = (
                build_helmholtz_2d_coo(
                    vp,
                    omega=omega,
                    dz=float(mesh.dz),
                    dx=float(mesh.dx),
                    boundary="pml",
                    pml_thickness=self.survey.pml_thickness,
                    pml_decay_factor=self.survey.pml_decay_factor,
                    pml_target_reflection=self.survey.pml_target_reflection,
                )
                if use_pml
                else build_helmholtz_2d_coo(
                    vp,
                    omega=omega,
                    dz=float(mesh.dz),
                    dx=float(mesh.dx),
                    abc=self.survey.abc,
                    n_pml=self.survey.n_pml,
                )
            )
            solved = solve_helmholtz_system(
                matrix,
                rhs,
                frequency_hz=frequency_hz,
                relative_tolerance=self.survey.residual_tolerance,
            )
            grid = solved.reshape(nz, nx, self.survey.n_shot)
            frequency_records.append(
                torch.stack(
                    [
                        grid[iz, ix, receiver.shot_id]
                        for (iz, ix), receiver in zip(receiver_indices, self.survey.receivers)
                    ]
                )
            )
            if self.survey.return_wavefield:
                wavefields.append(grid.permute(2, 0, 1))
        pressure = torch.stack(frequency_records, dim=1).unsqueeze(-1).contiguous()
        fields: dict[str, torch.Tensor] = {}
        if wavefields:
            fields["wavefield"] = torch.stack(wavefields, dim=0)
        method = "helmholtz_2d_pml" if use_pml else f"helmholtz_2d_{self.survey.abc}"
        return ForwardOutput(
            data={"p": pressure},
            fields=fields,
            metadata={
                "channels": ("p",),
                "units": {"p": "Pa"},
                "method": method,
                "domain": "frequency",
                "axis_names": ("trace", "frequency", "component"),
                "shape": ("n_trace", "n_frequency", "n_component"),
                "components": ("pressure",),
                "frequencies_hz": self.survey.frequencies,
                "frequencies": self.survey.frequencies,
                "source_shot_index": tuple(source.shot_id for source in self.survey.sources),
                "receiver_shot_index": tuple(receiver.shot_id for receiver in self.survey.receivers),
                "phasor_convention": "exp(+i*omega*t)",
                "source_normalization": "integrated-strength/cell-area",
                "solver": self.survey.solver,
                "n_pml": self.survey.n_pml,
                "pml_thickness": self.survey.pml_thickness,
                "pml_decay_factor": self.survey.pml_decay_factor,
                "pml_target_reflection": self.survey.pml_target_reflection,
            },
        )


__all__ = [
    "Helmholtz2D",
    "Helmholtz2DReceiver",
    "Helmholtz2DSource",
    "Helmholtz2DSurvey",
]
