"""FAMILIES: the single source of truth for GeoBrain's physics families.

Five self-contained physics families ship under ``geobrain/physics/``: each
owns its own data classes, solvers, and operator wrappers, and does not
import from the others (a family-isolation governance test enforces this).

This tuple is consumed by every piece of governance that needs to enumerate
"all the families", so adding, removing, or renaming a family is a
single-line edit here, and everything downstream (the isolation test, the
per-family advertised-surface freeze, the benchmark suite loader/dtype
policy, and the golden-case registry coverage check) picks it up
automatically instead of silently omitting the new family until someone
remembers to grep for the other hardcoded copies.

A completeness meta-test verifies this tuple actually stays in sync with
reality: it cross-checks
:data:`FAMILIES` against the physics filesystem, the benchmark suite modules,
and the golden-case registry, and fails loudly (naming exactly what is
missing where) the moment any of those four drift apart, e.g. a 6th family
directory lands without an entry here, or vice versa.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

FAMILIES: tuple[str, ...] = ("em", "flow", "potential", "rock", "wave")

__all__ = ["FAMILIES"]
