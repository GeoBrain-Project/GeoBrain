"""The shared earth model: the typed multi-field model layer.

``geobrain.geomodel.earthmodel`` lives inside ``geomodel``: geomodelling is
the activity and the shared earth model is its artifact, so the container
belongs under the one "model layer" door, ``from geobrain.geomodel import
EarthModel, Field, Link`` is the advertised spelling. The purity rule is
location-independent: this subpackage imports ``geobrain.core``
only, knows no physics operator, and the physics families are forbidden
from importing it (enforced by the architecture layer contracts). Three
classes:

- :class:`Field`: a single named unknown: either a plain (optionally
  bounded/transformed) tensor leaf, or an ``nn.Module`` generator.
- :class:`Link`: a differentiable derived field: ``fn(*args, **params)``
  wired into the model's dependency DAG.
- :class:`EarthModel`: binds a mesh + fields + links (+ optional frozen
  region pins) into a resolvable, invertible model:
  ``model.resolve()`` evaluates the DAG once; ``model.as_transform()``
  bridges it into any ``ForwardOperator`` chain
  (``physics_op @ model.as_transform()``) so the SAME ``Inverter`` /
  bayes-sampler machinery every other problem uses drives an EarthModel too.

Quick start::

    >>> import torch
    >>> from geobrain.core import TensorMesh
    >>> from geobrain.geomodel.earthmodel import EarthModel, Field
    >>> mesh = TensorMesh(shape=(4, 5), spacing=10.0)
    >>> model = EarthModel(mesh, fields={
    ...     "vp": Field(init=torch.full(mesh.shape, 2000.0), bounds=(500.0, 4500.0)),
    ... })
    >>> state = model.resolve()
    >>> sorted(state.tensors)
    ['vp']

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .field import Field
from .field_specs import FIELD_SPECS
from .link import Link
from .model import EarthModel

__all__ = ["EarthModel", "Field", "Link", "FIELD_SPECS"]
