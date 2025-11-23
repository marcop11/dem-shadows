# src/dem_shadows/analysis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from tqdm import tqdm

from .config import AnalysisConfig
from .utils import get_logger

logger = get_logger(__name__)


def list_shadow_files(folder: Path, pattern: str = "shadow_*.tif") -> List[Path]:
    """
    List shadow rasters to accumulate.

    For now this just glob-matches the pattern in the folder.
    You can refine later if you want date or time filtering.
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    return files


def compute_cumulative(cfg: AnalysisConfig) -> None:
    """
    Build a cumulative sun-exposure raster from a folder of shadow rasters.

    Assumes each input is uint8 with:
        0   = shadow
        1   = sunlit
        255 = nodata

    The output is uint16, where each pixel value is the number of frames
    where that pixel was sunlit (== 1) across all inputs.

    Notes
    -----
    - If a pixel is nodata in ALL input rasters, its sun-count will be 0.
      We write nodata=0 in the output profile, so you can't distinguish
      "never sunlit" from "always nodata" in the GeoTIFF alone. If you need
      that distinction, we can change to a different sentinel (e.g. 65535)
      and/or write a separate validity mask.
    """
    shadow_folder = cfg.shadow_folder
    out_path = cfg.out_path

    files = list_shadow_files(shadow_folder)
    if not files:
        raise RuntimeError(f"No shadow_*.tif files found in {shadow_folder}")

    logger.info("Found %d shadow rasters in %s", len(files), shadow_folder)

    # Open first file to define shape & profile
    with rasterio.open(files[0]) as src0:
        height, width = src0.height, src0.width
        profile = src0.profile.copy()
        nodata_val = src0.nodata

    # Check we won't overflow uint16
    if len(files) > cfg.max_value:
        raise RuntimeError(
            f"Too many rasters ({len(files)} > {cfg.max_value}) for uint16 "
            "sun-count; increase max_value and/or change dtype logic."
        )

    # Running sum of "sun" (== 1) counts
    sum_arr = np.zeros((height, width), dtype=np.uint16)

    # Track where we have any valid data at all
    valid_count = np.zeros((height, width), dtype=np.uint16)

    for fp in tqdm(files, desc="Accumulating sun counts", unit="raster"):
        with rasterio.open(fp) as src:
            # masked=True -> masks out src.nodata
            data = src.read(1, masked=True)

            # data is 0 (shadow), 1 (sun), or masked (outside DEM / nodata)
            # Fill masked with 0 so they don't contribute to the sun sum
            filled = data.filled(0).astype(sum_arr.dtype)

            # Add to running sum (only 1s contribute)
            sum_arr += filled

            # Count where data was valid (not masked)
            valid_count += (~data.mask).astype(valid_count.dtype)

    # Optional masked array if you want to inspect in Python
    sun_count = np.ma.array(sum_arr, mask=(valid_count == 0))
    logger.info(
        "Computed sun-count from %d rasters; min=%d, max=%d",
        len(files),
        int(sun_count.min()),
        int(sun_count.max()),
    )

    # Prepare output profile
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.uint16,
        count=1,
        nodata=0,  # 0 can mean 'never sunlit' or 'no data'
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(sum_arr, 1)

    logger.info("Wrote cumulative sun raster to %s", out_path)


def main_cli() -> None:
    """
    CLI entry point: dem-shadows-cumulate

    Example:

      dem-shadows-cumulate ^
        --shadow-folder "C:\\path\\to\\shadows" ^
        --out "C:\\path\\to\\sun_count.tif"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Accumulate sunlit counts from per-timestep shadow rasters."
    )
    parser.add_argument(
        "--shadow-folder",
        type=Path,
        required=True,
        help="Folder containing shadow_*.tif rasters.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output GeoTIFF path for cumulative sun count.",
    )

    args = parser.parse_args()

    cfg = AnalysisConfig(
        shadow_folder=args.shadow_folder,
        out_path=args.out,
        # You can adjust these defaults later if desired
        nodata=0,
        max_value=65535,
    )

    compute_cumulative(cfg)