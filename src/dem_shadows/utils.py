# src/dem_shadows/utils.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator, Tuple
import rasterio
from rasterio.warp import transform
from tzfpy import get_tz

def get_logger(name: str = "dem_shadows") -> logging.Logger:
    """
    Return a module-level logger with a sensible default format.
    Only configures the root handler once.
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger


def ensure_dir(path: Path) -> None:
    """
    Create directory (and parents) if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


@contextmanager
def time_block(label: str) -> Iterator[None]:
    """
    Simple timing context manager:

        with time_block("shadow batch"):
            run_shadow_batch(cfg)
    """
    logger = get_logger(__name__)
    start = perf_counter()
    logger.info("Starting %s...", label)
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("Finished %s in %.2f s", label, elapsed)


# Global logger for this module
logger = get_logger(__name__)


def get_dem_center_latlon(dem_path: Path) -> Tuple[float, float]:
    """
    Compute the geographic (lat, lon) of the DEM's center.

    Works even if the DEM is in a projected CRS (e.g. UTM).
    Returns (lat, lon) in EPSG:4326.

    Parameters
    ----------
    dem_path:
        Path to DEM GeoTIFF.

    Returns
    -------
    (lat, lon): tuple of floats

    Raises
    ------
    ValueError if DEM has no CRS.
    """
    dem_path = Path(dem_path)

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError(f"DEM {dem_path} has no CRS; cannot compute center lat/lon.")

        bounds = src.bounds
        # Center in DEM CRS (x, y)
        x_c = (bounds.left + bounds.right) / 2.0
        y_c = (bounds.bottom + bounds.top) / 2.0

        # Reproject single point to EPSG:4326 (lon/lat)
        lon_arr, lat_arr = transform(
            src.crs,
            "EPSG:4326",
            [x_c],
            [y_c],
        )

    lon = float(lon_arr[0])
    lat = float(lat_arr[0])

    logger.info("DEM center at lat=%.6f, lon=%.6f (EPSG:4326)", lat, lon)
    return lat, lon

def get_dem_timezone_tzfpy(dem_path: Path) -> str:
    """
    Determine IANA timezone name (e.g. 'Europe/Zurich') for a DEM
    by taking its center lat/lon and querying tzfpy.
    """
    lat, lon = get_dem_center_latlon(dem_path)
    tz = get_tz(lon, lat)  # tzfpy expects (lon, lat)
    if not tz:
        tz = "UTC"
        logger.warning(
            "tzfpy returned no timezone for lat=%.6f, lon=%.6f; falling back to UTC",
            lat,
            lon,
        )
    else:
        logger.info("Detected timezone %s for lat=%.6f, lon=%.6f", tz, lat, lon)
    return tz