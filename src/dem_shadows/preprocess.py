# src/dem_shadows/preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

from .utils import get_logger, ensure_dir

logger = get_logger(__name__)


def _check_same_crs_and_res(datasets: List[rasterio.io.DatasetReader]) -> None:
    """
    Check that all input rasters share the same CRS and pixel size.
    Raises ValueError if they don't.
    """
    ref = datasets[0]
    ref_crs = ref.crs
    # Pixel size from affine transform: a = pixel width, e = pixel height (usually negative)
    ref_res = (abs(ref.transform.a), abs(ref.transform.e))

    for src in datasets[1:]:
        if src.crs != ref_crs:
            raise ValueError(
                f"CRS mismatch between DEM tiles: {ref_crs} vs {src.crs} "
                f"({src.name})"
            )

        res = (abs(src.transform.a), abs(src.transform.e))
        if not np.allclose(res, ref_res, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Resolution mismatch between DEM tiles: {ref_res} vs {res} "
                f"({src.name})"
            )


def merge_dems(
    input_paths: Iterable[Path],
    out_path: Path,
    check_match: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Merge multiple DEM GeoTIFFs into a single GeoTIFF.

    Parameters
    ----------
    input_paths:
        Iterable of Paths to DEM tiles.
    out_path:
        Output GeoTIFF path for the merged DEM.
    check_match:
        If True, verify all inputs share the same CRS and pixel size.
        Raises ValueError if they differ.
    overwrite:
        If False and out_path exists, it is left untouched.

    Returns
    -------
    Path
        Path to the merged DEM.
    """
    input_paths = [Path(p) for p in input_paths]
    if not input_paths:
        raise ValueError("No input DEMs provided to merge_dems")

    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        logger.info("Merged DEM already exists at %s, skipping merge.", out_path)
        return out_path

    datasets = [rasterio.open(p) for p in input_paths]
    try:
        if check_match:
            _check_same_crs_and_res(datasets)

        logger.info("Merging %d DEM tiles into %s", len(datasets), out_path)
        mosaic, out_transform = rio_merge(datasets)

        meta = datasets[0].meta.copy()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
            }
        )

        ensure_dir(out_path.parent)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)

    finally:
        for ds in datasets:
            ds.close()

    return out_path


def merge_dems_in_folder(
    folder: Path,
    pattern: str = "*.tif",
    out_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Convenience wrapper: merge all DEM tiles in a folder.

    Parameters
    ----------
    folder:
        Directory containing DEM tiles.
    pattern:
        Glob pattern for DEMs (default: '*.tif').
    out_path:
        Output merged DEM. If None, writes 'merged_dem.tif' in the same folder.
    overwrite:
        If False and out_path exists, it is left untouched.

    Returns
    -------
    Path
        Path to the merged DEM.
    """
    folder = Path(folder)
    input_paths = sorted(folder.glob(pattern))
    if not input_paths:
        raise RuntimeError(f"No files matching {pattern!r} found in {folder}")

    if out_path is None:
        out_path = folder / "merged_dem.tif"

    return merge_dems(input_paths, out_path, overwrite=overwrite)