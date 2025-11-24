# src/dem_shadows/__init__.py

"""
DEM Shadows

Tools for computing terrain shadow rasters, cumulative sun/shadow maps,
and animations from DEM GeoTIFFs.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dem-shadows")
except PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.1.0"

# Re-export common user-facing functions and configs
from .config import LocationConfig, ShadowConfig, AnalysisConfig, AnimationConfig
from .shadows import run_shadow_batch
from .analysis import compute_cumulative
from .animate import create_shadow_gif
from .preprocess import merge_dems, merge_dems_in_folder

__all__ = [
    "LocationConfig",
    "ShadowConfig",
    "AnalysisConfig",
    "AnimationConfig",
    "run_shadow_batch",
    "compute_cumulative",
    "create_shadow_gif",
    "merge_dems",
    "merge_dems_in_folder",
    "__version__",
]