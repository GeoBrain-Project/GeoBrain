"""
Column role enum for :class:`GeoFrame`.

Roles tag the *semantic
meaning* of a property column (primary value vs. secondary covariate
vs. category indicator etc.); they let downstream operators
(kriging / cokriging / IK) discover their inputs without hard-coded
column names.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from enum import Enum

from ...core import GeoBrainError


class ColumnRole(str, Enum):
    """
    Semantic role of a :class:`GeoFrame` property column.

    String-valued so ``role == "value"`` works.
    """

    VALUE = "value"                    # primary measured variable (e.g. log-perm)
    SECONDARY = "secondary"            # covariate for cokriging
    CATEGORY = "category"              # integer category / facies indicator
    SELECTION = "selection"            # boolean mask
    WEIGHT = "weight"                  # declustering / observation weight
    ERROR_VARIANCE = "error_variance"  # heteroscedastic measurement variance
    VARIANCE = "variance"              # generic variance column
    ESTIMATE = "estimate"              # kriging estimate output
    SIMULATION = "simulation"          # simulation realisation output


def normalize_role(role: ColumnRole | str | None) -> ColumnRole | None:
    """
    Normalise a role spec into a :class:`ColumnRole` (or ``None``).

    Accepts the enum itself, the string value (case-insensitive), or
    ``None`` for "untagged". Raises :class:`GeoBrainError` on garbage.
    """
    if role is None:
        return None
    if isinstance(role, ColumnRole):
        return role
    if isinstance(role, str):
        try:
            return ColumnRole(role.lower())
        except ValueError as exc:
            valid = [r.value for r in ColumnRole]
            raise GeoBrainError(
                f"unknown column role {role!r}",
                object_name="normalize_role", field="role",
                expected=f"one of {valid}", actual=role,
            ) from exc
    raise GeoBrainError(
        "role must be a ColumnRole, string, or None",
        object_name="normalize_role", field="role",
        expected="ColumnRole | str | None", actual=type(role).__name__,
    )


__all__ = ["ColumnRole", "normalize_role"]
