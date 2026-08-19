"""
Rock-physics model registry with category inference + aliases.

Thin wrapper around :class:`geobrain.core.registry.Registry` adding:

- **Category inference** from the model's base class
  (``EffectiveModel`` → category ``"effective"``, etc.). Saves boilerplate
  on every ``@register`` decoration.
- **Alias resolution** (``"vrh"`` → ``"VRH"``). Rock-physics literature
  uses many short / capitalisation variants for each model.
- **Metadata** stored alongside the class for filtering / docs.

Use via the :func:`register` decorator:

    @register("VRH", aliases=["vrh", "voigt_reuss_hill"], description="...")
    class VRH(EffectiveModel):
        ...

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any

from geobrain.core.registry import Registry

from ._base import BaseModel
from ._categories import (
    AnisotropyModel,
    EffectiveModel,
    EmpiricalModel,
    FluidModel,
    GranularModel,
    PermeabilityModel,
    ResistivityModel,
)

_registry: Registry[type[BaseModel]] = Registry("rock_models")
_aliases: dict[str, str] = {}
_metadata: dict[str, dict[str, Any]] = {}


_CATEGORY_BY_BASE: tuple[tuple[type, str], ...] = (
    (EffectiveModel, "effective"),
    (FluidModel, "fluid"),
    (GranularModel, "granular"),
    (EmpiricalModel, "empirical"),
    (AnisotropyModel, "anisotropy"),
    (PermeabilityModel, "permeability"),
    (ResistivityModel, "resistivity"),
)


def _infer_category(cls: type[BaseModel]) -> str:
    for base, label in _CATEGORY_BY_BASE:
        if issubclass(cls, base):
            return label
    return "other"


def register(
    name: str | None = None,
    *,
    category: str | None = None,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    **metadata,
):
    """
    Decorator: register a :class:`BaseModel` subclass.

    Args:
        name: Canonical name. Defaults to ``cls.__name__``.
        category: Override the inferred category.
        aliases: Alternative names (lookup-only, not canonical).
        tags: Free-form tags for filtering.
        **metadata: Extra keys stored in the model's metadata dict.

    Re-registration under the same canonical name overwrites the entry
    (so module reloads in test sessions don't blow up).
    """

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        canonical = name or cls.__name__
        # overwrite=True is the public seam Registry already exposes for this
        # exact need (reload idempotency): no reason to reach into the
        # private _entries dict for it.
        _registry.register(canonical, overwrite=True)(cls)
        _metadata[canonical] = {
            "category": category or _infer_category(cls),
            "tags": list(tags or ()),
            **metadata,
        }
        for alias in aliases or ():
            # An alias may not hijack another model's canonical name, and an
            # already-bound alias may not be silently re-pointed: either
            # would make lookups depend on registration order.
            if alias in _metadata and alias != canonical:
                raise ValueError(
                    f"rock model alias {alias!r} collides with the canonical "
                    f"name of another registered model"
                )
            existing = _aliases.get(alias)
            if existing is not None and existing != canonical:
                raise ValueError(
                    f"rock model alias {alias!r} is already bound to "
                    f"{existing!r}; refusing to re-point it to {canonical!r}"
                )
            _aliases[alias] = canonical
        return cls

    return decorator


def _resolve(name: str) -> str:
    return _aliases.get(name, name)


def get_model(name: str) -> type[BaseModel]:
    """Look up a model class by canonical name or alias."""
    return _registry.get(_resolve(name))


def create_model(name: str, **kwargs) -> BaseModel:
    """Instantiate a model by name (canonical or alias)."""
    return get_model(name)(**kwargs)


def list_models(category: str | None = None) -> list[str]:
    """List canonical names; optionally filter by category."""
    names = list(_registry.names())
    if category is None:
        return names
    return [n for n in names if _metadata.get(n, {}).get("category") == category]


def list_categories() -> list[str]:
    """Return the set of categories actually represented in the registry."""
    return sorted({m.get("category", "other") for m in _metadata.values()})


def list_aliases() -> dict[str, str]:
    """Alias → canonical name map."""
    return dict(_aliases)


def get_model_metadata(name: str) -> dict[str, Any]:
    """Return the metadata dict for a registered model."""
    return dict(_metadata.get(_resolve(name), {}))


__all__ = [
    "create_model",
    "get_model",
    "get_model_metadata",
    "list_aliases",
    "list_categories",
    "list_models",
    "register",
]
