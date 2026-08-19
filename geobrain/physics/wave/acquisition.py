"""Packed source/receiver acquisition geometry for Wave physics.

The canonical surveys hold physical coordinates in public order ``(x, z)``
for 2-D and ``(x, y, z)`` for 3-D.  Each source and receiver trace carries a
contiguous shot identifier, permitting irregular receiver layouts and multiple
sources within one shot without a parallel legacy representation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from collections.abc import Iterable
from typing import ClassVar, SupportsFloat, SupportsIndex, TypeVar, cast

import torch

from .errors import WaveContractError


def _contract_error(
    object_name: str, field: str, expected: object, actual: object,
) -> WaveContractError:
    """Build a consistently attributed public-contract error."""
    return WaveContractError(
        f"{object_name}.{field} must be {expected}; got {actual}",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _factory_metadata(
    value: object,
    *,
    dtype: torch.dtype,
    object_name: str,
    field: str,
) -> torch.Tensor:
    """Convert non-Tensor factory metadata, retaining Tensor inputs exactly."""
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value, dtype=dtype, device="cpu")
    except (TypeError, ValueError, OverflowError) as exc:
        raise _contract_error(
            object_name,
            field,
            f"CPU {dtype} Tensor-convertible metadata",
            value,
        ) from exc


def _validate_tensor(
    value: object,
    *,
    object_name: str,
    field: str,
    shape: tuple[int | None, ...],
    dtype: torch.dtype,
    positions: bool = False,
) -> torch.Tensor:
    """Validate exact Tensor metadata and return a detached owned CPU clone."""
    if not isinstance(value, torch.Tensor):
        raise _contract_error(object_name, field, "a torch.Tensor", type(value))
    if value.device.type != "cpu":
        raise _contract_error(object_name, field, "a CPU Tensor", value.device)
    if value.dtype is not dtype:
        raise _contract_error(object_name, field, dtype, value.dtype)
    if value.ndim != len(shape) or any(
        actual != expected for actual, expected in zip(value.shape, shape) if expected is not None
    ):
        raise _contract_error(object_name, field, f"shape {shape}", tuple(value.shape))
    if positions and value.requires_grad:
        raise _contract_error(object_name, field, "a Tensor without requires_grad", "requires_grad=True")
    if positions and not bool(torch.isfinite(value).all()):
        raise _contract_error(object_name, field, "finite values", "non-finite values")
    return value.detach().clone()


def _validate_shot_ids(
    source_shot_index: torch.Tensor,
    receiver_shot_index: torch.Tensor,
    *,
    object_name: str,
) -> int:
    """Require paired, non-empty canonical shot domains ``0..n_shot-1``."""
    source_ids = torch.unique(source_shot_index, sorted=True)
    receiver_ids = torch.unique(receiver_shot_index, sorted=True)
    for field, ids in (
        ("source_shot_index", source_ids),
        ("receiver_shot_index", receiver_ids),
    ):
        if ids.numel() == 0:
            raise _contract_error(object_name, field, "at least one shot", "empty")
        if int(ids[0]) < 0:
            raise _contract_error(object_name, field, "non-negative shot identifiers", ids.tolist())
        canonical = torch.arange(ids.numel(), dtype=torch.int64)
        if not torch.equal(ids, canonical):
            raise _contract_error(object_name, field, "contiguous shot IDs 0..n_shot-1", ids.tolist())
    if not torch.equal(source_ids, receiver_ids):
        raise _contract_error(
            object_name,
            "source_shot_index/receiver_shot_index",
            "matching source and receiver shot domains",
            (source_ids.tolist(), receiver_ids.tolist()),
        )
    return int(source_ids.numel())


def _validate_packed_survey(
    *,
    ndim: int,
    object_name: str,
    source_positions: object,
    source_shot_index: object,
    receiver_positions: object,
    receiver_shot_index: object,
    nt: object,
    dt: object,
    t0: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
    """Validate and own the dimension-parametric canonical survey metadata."""
    source_positions_tensor = _validate_tensor(
        source_positions,
        object_name=object_name,
        field="source_positions",
        shape=(None, ndim),
        dtype=torch.float64,
        positions=True,
    )
    receiver_positions_tensor = _validate_tensor(
        receiver_positions,
        object_name=object_name,
        field="receiver_positions",
        shape=(None, ndim),
        dtype=torch.float64,
        positions=True,
    )
    source_shot_index_tensor = _validate_tensor(
        source_shot_index,
        object_name=object_name,
        field="source_shot_index",
        shape=(source_positions_tensor.shape[0],),
        dtype=torch.int64,
    )
    receiver_shot_index_tensor = _validate_tensor(
        receiver_shot_index,
        object_name=object_name,
        field="receiver_shot_index",
        shape=(receiver_positions_tensor.shape[0],),
        dtype=torch.int64,
    )
    _validate_shot_ids(
        source_shot_index_tensor,
        receiver_shot_index_tensor,
        object_name=object_name,
    )
    if isinstance(nt, bool) or not isinstance(nt, int) or nt <= 0:
        raise _contract_error(object_name, "nt", "a positive int", nt)
    dt_value = _finite_scalar(dt, object_name=object_name, field="dt", positive=True)
    t0_value = _finite_scalar(t0, object_name=object_name, field="t0", positive=False)
    return (
        source_positions_tensor,
        source_shot_index_tensor,
        receiver_positions_tensor,
        receiver_shot_index_tensor,
        int(nt),
        dt_value,
        t0_value,
    )


def _finite_scalar(
    value: object,
    *,
    object_name: str,
    field: str,
    positive: bool,
) -> float:
    """Validate an exactly representable finite scalar without overflow leaks."""
    expected = "a positive finite float" if positive else "a finite float"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _contract_error(object_name, field, expected, value)
    try:
        result = float(value)
    except OverflowError as exc:
        raise _contract_error(object_name, field, expected, value) from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise _contract_error(object_name, field, expected, value)
    return result


def _grid_coordinates(
    indices: object,
    *,
    ndim: int,
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    field: str,
    object_name: str,
) -> torch.Tensor:
    """Map platform-order grid indices to public cell-centre coordinates."""
    tensor = _factory_metadata(
        indices,
        dtype=torch.int64,
        object_name=object_name,
        field=field,
    )
    if isinstance(indices, torch.Tensor):
        tensor = _validate_tensor(
            tensor,
            object_name=object_name,
            field=field,
            shape=(None, ndim),
            dtype=torch.int64,
        )
    elif tensor.ndim != 2 or tensor.shape[1] != ndim:
        raise _contract_error(object_name, field, f"shape (n, {ndim})", tuple(tensor.shape))
    platform = tensor.to(torch.float64)
    spacing_tensor = torch.tensor(spacing, dtype=torch.float64)
    origin_tensor = torch.tensor(origin, dtype=torch.float64)
    coordinates = origin_tensor + (platform + 0.5) * spacing_tensor
    return coordinates[:, (1, 0)] if ndim == 2 else coordinates[:, (1, 2, 0)]


def _grid_parameters(
    *,
    ndim: int,
    spacing: object,
    origin: object | None,
    object_name: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate platform-order grid spacing and origin metadata."""
    try:
        spacing_tuple = tuple(
            float(cast(str | SupportsFloat | SupportsIndex, value))
            for value in cast(Iterable[object], spacing)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _contract_error(object_name, "spacing", f"{ndim} finite positive values", spacing) from exc
    if len(spacing_tuple) != ndim or any(not math.isfinite(value) or value <= 0 for value in spacing_tuple):
        raise _contract_error(object_name, "spacing", f"{ndim} finite positive values", spacing)
    if origin is None:
        return spacing_tuple, (0.0,) * ndim
    try:
        origin_tuple = tuple(
            float(cast(str | SupportsFloat | SupportsIndex, value))
            for value in cast(Iterable[object], origin)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _contract_error(object_name, "origin", f"{ndim} finite values", origin) from exc
    if len(origin_tuple) != ndim or any(not math.isfinite(value) for value in origin_tuple):
        raise _contract_error(object_name, "origin", f"{ndim} finite values", origin)
    return spacing_tuple, origin_tuple


_PackedSurveyT = TypeVar("_PackedSurveyT", bound="_PackedSurvey")


class _PackedSurvey:
    """Shared observable behavior of the two fixed public survey signatures."""

    __slots__ = ()

    _ndim: ClassVar[int]
    source_positions: torch.Tensor
    source_shot_index: torch.Tensor
    receiver_positions: torch.Tensor
    receiver_shot_index: torch.Tensor
    nt: int
    dt: float
    t0: float

    def __init__(
        self,
        source_positions: torch.Tensor,
        source_shot_index: torch.Tensor,
        receiver_positions: torch.Tensor,
        receiver_shot_index: torch.Tensor,
        *,
        nt: int,
        dt: float,
        t0: float = 0.0,
    ) -> None:
        """Declare the fixed constructor shared by concrete packed surveys."""
        raise NotImplementedError

    @property
    def n_source(self) -> int:
        """Number of packed source rows."""
        return int(self.source_positions.shape[0])

    @property
    def n_trace(self) -> int:
        """Number of packed receiver-trace rows."""
        return int(self.receiver_positions.shape[0])

    @property
    def n_shot(self) -> int:
        """Number of contiguous shots represented by the survey."""
        return int(torch.unique(self.source_shot_index).numel())

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of the canonical acquisition metadata."""
        digest = hashlib.sha256()
        for label, tensor in (
            (b"source_positions", self.source_positions),
            (b"source_shot_index", self.source_shot_index),
            (b"receiver_positions", self.receiver_positions),
            (b"receiver_shot_index", self.receiver_shot_index),
        ):
            contiguous = tensor.contiguous()
            digest.update(label)
            digest.update(struct.pack("<q", contiguous.ndim))
            digest.update(struct.pack("<" + "q" * contiguous.ndim, *contiguous.shape))
            digest.update(contiguous.numpy().tobytes())
        digest.update(b"nt")
        digest.update(_length_prefixed_nonnegative_int(self.nt))
        digest.update(b"dt")
        digest.update(struct.pack("<d", self.dt))
        digest.update(b"t0")
        digest.update(struct.pack("<d", self.t0))
        return digest.hexdigest()

    def to_dense(self, data: torch.Tensor, *, fill_value: object = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shot-major dense traces and an on-device validity mask."""
        if not isinstance(data, torch.Tensor):
            raise _contract_error(type(self).__name__, "data", "a torch.Tensor", type(data))
        if data.ndim != 3 or data.shape[0] != self.n_trace or data.shape[1] != self.nt:
            raise _contract_error(
                type(self).__name__,
                "data",
                f"shape ({self.n_trace}, {self.nt}, n_component)",
                tuple(data.shape),
            )
        counts = torch.bincount(self.receiver_shot_index, minlength=self.n_shot)
        width = int(counts.max())
        dense = data.new_full((self.n_shot, self.nt, width, data.shape[2]), fill_value)
        mask = torch.zeros((self.n_shot, width), dtype=torch.bool, device=data.device)
        for shot in range(self.n_shot):
            indices = torch.nonzero(self.receiver_shot_index == shot, as_tuple=False).flatten()
            count = int(indices.numel())
            trace_indices = indices.to(data.device)
            dense[shot, :, :count, :] = data[trace_indices].permute(1, 0, 2)
            mask[shot, :count] = True
        return dense, mask

    @classmethod
    def from_positions(
        cls: type[_PackedSurveyT],
        source_positions: object,
        source_shot_index: object,
        receiver_positions: object,
        receiver_shot_index: object,
        *,
        nt: int,
        dt: float,
        t0: float = 0.0,
    ) -> _PackedSurveyT:
        """Build a packed survey, converting only non-Tensor factory metadata."""
        return cls(
            _factory_metadata(
                source_positions,
                dtype=torch.float64,
                object_name=cls.__name__,
                field="source_positions",
            ),
            _factory_metadata(
                source_shot_index,
                dtype=torch.int64,
                object_name=cls.__name__,
                field="source_shot_index",
            ),
            _factory_metadata(
                receiver_positions,
                dtype=torch.float64,
                object_name=cls.__name__,
                field="receiver_positions",
            ),
            _factory_metadata(
                receiver_shot_index,
                dtype=torch.int64,
                object_name=cls.__name__,
                field="receiver_shot_index",
            ),
            nt=nt,
            dt=dt,
            t0=t0,
        )

    @classmethod
    def from_shared_receivers(
        cls: type[_PackedSurveyT],
        source_positions: object,
        receiver_positions: object,
        *,
        nt: int,
        dt: float,
        t0: float = 0.0,
    ) -> _PackedSurveyT:
        """Build one-source-per-shot geometry with a repeated receiver set."""
        source = _factory_metadata(
            source_positions,
            dtype=torch.float64,
            object_name=cls.__name__,
            field="source_positions",
        )
        receiver = _factory_metadata(
            receiver_positions,
            dtype=torch.float64,
            object_name=cls.__name__,
            field="receiver_positions",
        )
        if source.ndim != 2 or source.shape[1] != cls._ndim:
            raise _contract_error(cls.__name__, "source_positions", f"shape (n, {cls._ndim})", tuple(source.shape))
        if receiver.ndim != 2 or receiver.shape[1] != cls._ndim:
            raise _contract_error(cls.__name__, "receiver_positions", f"shape (n, {cls._ndim})", tuple(receiver.shape))
        n_source = source.shape[0]
        source_shots = torch.arange(n_source, dtype=torch.int64)
        receiver_shots = torch.arange(n_source, dtype=torch.int64).repeat_interleave(receiver.shape[0])
        return cls.from_positions(
            source,
            source_shots,
            receiver.repeat((n_source, 1)),
            receiver_shots,
            nt=nt,
            dt=dt,
            t0=t0,
        )

    @classmethod
    def from_grid_indices(
        cls: type[_PackedSurveyT],
        source_indices: object,
        source_shot_index: object,
        receiver_indices: object,
        receiver_shot_index: object,
        *,
        spacing: object,
        origin: object | None = None,
        nt: int,
        dt: float,
        t0: float = 0.0,
    ) -> _PackedSurveyT:
        """Build a survey from platform-order grid indices at cell centres."""
        spacing_values, origin_values = _grid_parameters(
            ndim=cls._ndim, spacing=spacing, origin=origin, object_name=cls.__name__,
        )
        return cls.from_positions(
            _grid_coordinates(
                source_indices,
                ndim=cls._ndim,
                spacing=spacing_values,
                origin=origin_values,
                field="source_indices",
                object_name=cls.__name__,
            ),
            source_shot_index,
            _grid_coordinates(
                receiver_indices,
                ndim=cls._ndim,
                spacing=spacing_values,
                origin=origin_values,
                field="receiver_indices",
                object_name=cls.__name__,
            ),
            receiver_shot_index,
            nt=nt,
            dt=dt,
            t0=t0,
        )


@dataclass(frozen=True, slots=True)
class Seismic2DSurvey(_PackedSurvey):
    """Packed 2-D source/receiver acquisition with public ``(x, z)`` positions.

    Position columns are world order ``(x, z)`` metres with z the platform
    DEPTH axis (positive down): the engine maps column 0 → mesh axis-1 (x)
    and column 1 → axis-0 (z). :meth:`from_grid_indices` instead takes
    platform-order ``(iz, ix)`` cell indices.

    Attributes:
        source_positions: ``(n_src, 2)`` source ``(x, z)`` [m].
        source_shot_index: shot id per source row.
        receiver_positions: ``(n_rcv, 2)`` receiver ``(x, z)`` [m].
        receiver_shot_index: shot id per receiver row.
        nt / dt / t0: record length [samples], sample interval [s], start
            time [s].
    """

    source_positions: torch.Tensor
    source_shot_index: torch.Tensor
    receiver_positions: torch.Tensor
    receiver_shot_index: torch.Tensor
    nt: int
    dt: float
    t0: float = 0.0

    _ndim: ClassVar[int] = 2

    def __post_init__(self) -> None:
        values = _validate_packed_survey(
            ndim=self._ndim,
            object_name=type(self).__name__,
            source_positions=self.source_positions,
            source_shot_index=self.source_shot_index,
            receiver_positions=self.receiver_positions,
            receiver_shot_index=self.receiver_shot_index,
            nt=self.nt,
            dt=self.dt,
            t0=self.t0,
        )
        for field, value in zip(
            ("source_positions", "source_shot_index", "receiver_positions", "receiver_shot_index", "nt", "dt", "t0"),
            values,
        ):
            object.__setattr__(self, field, value)


@dataclass(frozen=True, slots=True)
class Seismic3DSurvey(_PackedSurvey):
    """Packed 3-D source/receiver acquisition with public ``(x, y, z)`` positions.

    Position columns are world order ``(x, y, z)`` metres with z the
    platform DEPTH axis (positive down): the engine maps x → mesh axis-1,
    y → axis-2, z → axis-0. :meth:`from_grid_indices` instead takes
    platform-order ``(iz, ix, iy)`` cell indices.

    Attributes:
        source_positions: ``(n_src, 3)`` source ``(x, y, z)`` [m].
        source_shot_index: shot id per source row.
        receiver_positions: ``(n_rcv, 3)`` receiver ``(x, y, z)`` [m].
        receiver_shot_index: shot id per receiver row.
        nt / dt / t0: record length [samples], sample interval [s], start
            time [s].
    """

    source_positions: torch.Tensor
    source_shot_index: torch.Tensor
    receiver_positions: torch.Tensor
    receiver_shot_index: torch.Tensor
    nt: int
    dt: float
    t0: float = 0.0

    _ndim: ClassVar[int] = 3

    def __post_init__(self) -> None:
        values = _validate_packed_survey(
            ndim=self._ndim,
            object_name=type(self).__name__,
            source_positions=self.source_positions,
            source_shot_index=self.source_shot_index,
            receiver_positions=self.receiver_positions,
            receiver_shot_index=self.receiver_shot_index,
            nt=self.nt,
            dt=self.dt,
            t0=self.t0,
        )
        for field, value in zip(
            ("source_positions", "source_shot_index", "receiver_positions", "receiver_shot_index", "nt", "dt", "t0"),
            values,
        ):
            object.__setattr__(self, field, value)


def shared_wavelet(wavelet: torch.Tensor, *, n_source: int) -> torch.Tensor:
    """Expand a shared one-dimensional wavelet into an owned source matrix.

    Args:
        wavelet: single 1-D source time function ``(nt,)``.
        n_source: number of shots to broadcast it to.
    """
    if not isinstance(wavelet, torch.Tensor):
        raise _contract_error("shared_wavelet", "wavelet", "a torch.Tensor", type(wavelet))
    if wavelet.ndim != 1:
        raise _contract_error("shared_wavelet", "wavelet", "shape (nt,)", tuple(wavelet.shape))
    if isinstance(n_source, bool) or not isinstance(n_source, int) or n_source <= 0:
        raise _contract_error("shared_wavelet", "n_source", "a positive int", n_source)
    return wavelet.unsqueeze(0).expand(n_source, -1).clone()


def _length_prefixed_nonnegative_int(value: int) -> bytes:
    """Encode an arbitrary-size non-negative integer without a fixed-width cast."""
    width = max(1, (value.bit_length() + 7) // 8)
    encoded = value.to_bytes(width, byteorder="little", signed=False)
    return len(encoded).to_bytes(8, byteorder="little", signed=False) + encoded


__all__ = ["Seismic2DSurvey", "Seismic3DSurvey", "shared_wavelet"]
