"""
Synthetic-approximation benchmark datasets.

Each loader returns a :class:`GeoFrame` whose properties and shapes
mirror the published versions, but whose values come from seeded
RNG processes rather than digitised source data. The loaders are
deterministic, repeated calls yield identical tables.

Public loaders:

- :func:`walker_lake`: 470 points, V & U columns.
- :func:`jura`: 359 points, 7 metals + Rock class.
- :func:`meuse`: 155 points, 4 metals + ``dist_river``.
- :func:`mining_3d`: 200 3-D drillhole points, Au column.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .jura import jura
from .meuse import meuse
from .mining_3d import mining_3d
from .walker_lake import walker_lake

__all__ = ["jura", "meuse", "mining_3d", "walker_lake"]
