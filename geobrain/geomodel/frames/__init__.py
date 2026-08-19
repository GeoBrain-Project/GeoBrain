"""
Spatial data containers for the geomodel subpackage.

All containers store coordinates as numpy arrays and validate via
:class:`GeoBrainError`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .geometry import Geometry
from .geo_points import GeoPoints
from .geo_grid import GeoGrid
from .gslib_layout import GslibGridLayout, gslib_grid_layout
from .metadata import Category, PropertyKind, PropertyMetadata
from .roles import ColumnRole, normalize_role
from .geo_frame import GeoFrame

__all__ = [
    "Geometry",
    "GeoPoints",
    "GeoGrid",
    "GslibGridLayout",
    "GeoFrame",
    "Category",
    "ColumnRole",
    "PropertyKind",
    "PropertyMetadata",
    "gslib_grid_layout",
    "normalize_role",
]
