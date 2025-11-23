# src/dem_shadows/shadows.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import rasterio
from zoneinfo import ZoneInfo
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from tqdm import tqdm
from .config import ShadowConfig
from .schedule import build_schedule
from .preprocess import merge_dems_in_folder
from .utils import get_logger, get_dem_center_latlon, get_dem_timezone_tzfpy

logger = get_logger(__name__)

def prepare_metric_dem(cfg: ShadowConfig) -> Path:
    """
    Ensure DEM is in a projected metric CRS.
    Returns path to metric DEM (may be original if already metric).
    """
    dem_path = cfg.dem_path
    metric_path = cfg.dem_metric_path or dem_path.with_name(dem_path.stem + "_metric.tif")

    with rasterio.open(dem_path) as src:
        if src.crs and src.crs.is_projected:
            # Already projected; just reuse
            return dem_path

        # Guess UTM zone from DEM centroid if not provided
        bounds = src.bounds
        lon_c = (bounds.left + bounds.right) / 2
        lat_c = (bounds.top + bounds.bottom) / 2
        if cfg.metric_crs:
            target_crs = cfg.metric_crs
        else:
            zone = int((lon_c + 180) // 6) + 1
            epsg = 32600 + zone if lat_c >= 0 else 32700 + zone
            target_crs = f"EPSG:{epsg}"

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(metric_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )

    return metric_path


def compute_shadow_for_time(
    dem_path: Path,
    timestamp,
    out_path: Path,
    overwrite: bool = False,
) -> None:
    """
    Compute a single shadow raster for a given timestamp using the
    `insolation` library (sunvector + doshade).

    Output:
        uint8 GeoTIFF
            1 = sunlit
            0 = shadow
            255 = nodata
    """
    if out_path.exists() and not overwrite:
        return

    from insolation.insolf import julian_day, sunvector, doshade
    from rasterio.windows import Window

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        nodata = src.nodata
        profile = src.profile.copy()

        # Resolution (assumes square pixels)
        res_x = src.transform.a
        res_y = -src.transform.e
        res = float((abs(res_x) + abs(res_y)) / 2.0)

        # Identify nodata mask
        if nodata is not None:
            dem_invalid = (dem == nodata)
        else:
            dem_invalid = np.zeros_like(dem, dtype=bool)

        # Compute DEM centroid (in DEM CRS)
        bounds = src.bounds
        lon_c, lat_c = src.lnglat()  # method exists only in Rasterio >= 1.4

        # Sun vector (timestamp is UTC)
        jd = julian_day(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
        )
        sun_v = sunvector(jd, latitude=lat_c, longitude=lon_c, timezone=0)

        # Compute boolean shadow map
        shadow_mask = doshade(
            dem.astype("float64"),
            res,
            sun_v,
            num_sweeps=1,
        )

        # Convert to uint8 classification
        #     1 = sunlit
        #     0 = shadow
        #   255 = nodata
        out_arr = shadow_mask.astype(np.uint8)
        out_arr[dem_invalid] = 255

        # Update profile
        profile.update(dtype=rasterio.uint8, nodata=255, count=1)

        # Write in chunks
        rows, cols = out_arr.shape
        chunk_rows = 2048

        with rasterio.open(out_path, "w", **profile) as dst:
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                window = Window(
                    col_off=0,
                    row_off=r0,
                    width=cols,
                    height=r1 - r0,
                )
                dst.write(out_arr[r0:r1, :], 1, window=window)


def shadow_filename_from_time(
    out_dir: Path,
    timestamp,
    timezone_name: str | None = None,
) -> Path:
    """
    Build filename like: shadow_YYYYMMDDThhmmss_TZ.tif

    - timestamp is expected in UTC
    - if timezone_name is given, convert to that tz for naming
    - TZ is the IANA name with '/' removed, e.g. 'EuropeZurich'
    """
    if timezone_name is not None:
        local_tz = ZoneInfo(timezone_name)
        timestamp = timestamp.astimezone(local_tz)

    stamp = timestamp.strftime("%Y%m%dT%H%M%S")  # YYYYMMDDThhmmss, no separators
    tz_suffix = ""
    if timezone_name is not None:
        tz_suffix = "_" + timezone_name.replace("/", "")

    name = f"shadow_{stamp}{tz_suffix}.tif"
    return out_dir / name


def run_shadow_batch(cfg: ShadowConfig) -> None:
    """
    High-level: build schedule, prepare DEM, run all shadow frames.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    metric_dem = prepare_metric_dem(cfg)
    schedule = build_schedule(cfg)

    # Optionally write schedule CSV
    if cfg.write_schedule_csv:
        csv_path = cfg.out_dir / cfg.schedule_csv_name
        from csv import writer
        with csv_path.open("w", newline="") as f:
            w = writer(f)
            w.writerow(["iso_utc"])
            for t in schedule:
                w.writerow([t.isoformat()])

    for t in tqdm(schedule, desc="Computing shadows"):
        out_path = shadow_filename_from_time(
            cfg.out_dir,
            t,
            timezone_name=cfg.location.timezone,  # use local time for naming
        )
        compute_shadow_for_time(metric_dem, t, out_path, overwrite=cfg.overwrite)


def main_cli():
    """
    Example CLI:

      # Fully automatic: DEM center + tzfpy timezone
      dem-shadows-generate --dem-dir "C:\\path\\to\\dem_tiles" --start 2024-06-01 --end 2024-06-02

      # Manual lat/lon, auto timezone
      dem-shadows-generate --dem my_dem.tif --lat 47.5 --lon 8.5 --start 2024-06-01 --end 2024-06-02

      # Manual lat/lon and manual timezone
      dem-shadows-generate --dem my_dem.tif --lat 47.5 --lon 8.5 --timezone Europe/Zurich --start 2024-06-01 --end 2024-06-02
    """
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Run DEM shadow batch.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dem", type=Path, help="Path to single DEM GeoTIFF")
    group.add_argument("--dem-dir", type=Path, help="Folder with DEM tiles to merge")

    parser.add_argument(
        "--dem-pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for DEM tiles (used with --dem-dir)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)

    # Lat/lon: optional; if omitted, we auto-use DEM center
    parser.add_argument("--lat", type=float, help="Latitude in decimal degrees")
    parser.add_argument("--lon", type=float, help="Longitude in decimal degrees")

    # Old flags kept for backwards compatibility (no longer necessary)
    parser.add_argument(
        "--auto-latlon",
        action="store_true",
        help="(Deprecated) Automatically use DEM center as lat/lon. This is now the default when --lat/--lon are omitted.",
    )

    # Timezone: optional; if omitted, we auto-detect via tzfpy
    parser.add_argument(
        "--timezone",
        type=str,
        default=None,
        help="Timezone name (e.g. 'Europe/Zurich'). If not set, auto-detect from DEM center.",
    )
    parser.add_argument(
        "--auto-timezone",
        action="store_true",
        help="(Deprecated) Automatically detect timezone from DEM center. This is now the default when --timezone is omitted.",
    )

    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=60,
        help="Interval between shadow frames in minutes (default: 60). Ignored if --only-time is used.",
    )
    parser.add_argument(
        "--only-time",
        type=str,
        default=None,
        help="Optional time-of-day (HH:MM, local) to run once per day, e.g. '10:15'. If set, ignores --step-minutes.",
    )

    args = parser.parse_args()

    # 1) Determine DEM path (single file or merged folder)
    if args.dem_dir is not None:
        merged_path = args.dem_dir / "merged_dem.tif"
        logger.info("Merging DEM tiles in %s", args.dem_dir)
        dem_path = merge_dems_in_folder(
            folder=args.dem_dir,
            pattern=args.dem_pattern,
            out_path=merged_path,
            overwrite=False,
        )
    else:
        dem_path = args.dem

    # 2) Determine latitude/longitude
    if args.lat is not None and args.lon is not None:
        # User explicitly provided coordinates
        lat, lon = args.lat, args.lon
    else:
        # Default: auto from DEM center
        lat, lon = get_dem_center_latlon(dem_path)

    # 3) Determine timezone
    if args.timezone is not None:
        # Manual timezone wins
        tz_name = args.timezone
    else:
        # Default: auto from DEM center via tzfpy
        tz_name = get_dem_timezone_tzfpy(dem_path)

    from .config import LocationConfig, ShadowConfig

    cfg = ShadowConfig(
        dem_path=dem_path,
        out_dir=args.out_dir,
        location=LocationConfig(
            latitude=lat,
            longitude=lon,
            timezone=tz_name,
        ),
        start_date=datetime.strptime(args.start, "%Y-%m-%d").date(),
        end_date=datetime.strptime(args.end, "%Y-%m-%d").date(),
        step_minutes=args.step_minutes,
        only_time=args.only_time,
    )

    run_shadow_batch(cfg)