# src/dem_shadows/config.py
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

@dataclass
class LocationConfig:
    latitude: float
    longitude: float
    timezone: str = "UTC"   # e.g. "Europe/Zurich"

@dataclass
class ShadowConfig:
    dem_path: Path                 # input DEM GeoTIFF
    out_dir: Path                  # where to write shadow rasters
    location: LocationConfig
    start_date: date
    end_date: date
    step_minutes: int = 60
    only_time: Optional[str] = None  # e.g. "10:15"
    twilight_depression: float = 6.0  # civil = 6°, nautical 12°, etc.
    overwrite: bool = False
    write_schedule_csv: bool = True
    schedule_csv_name: str = "schedule.csv"
    # DEM / reprojection options
    metric_crs: Optional[str] = None  # if None, guess UTM based on DEM centroid
    dem_metric_path: Optional[Path] = None

@dataclass
class AnalysisConfig:
    shadow_folder: Path      # dir with shadow_*.tif
    out_path: Path           # resulting cumulative map
    nodata: int = 255        # as in your shadow rasters
    max_value: int = 65535   # for uint16 accumulation

@dataclass
class AnimationConfig:
    shadow_folder: Path
    out_gif: Path
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    hour_filter: Optional[int] = None
    minute_filter: Optional[int] = None
    duration_ms: int = 250
    sample_stride: int = 1
    target_width: Optional[int] = 900
    target_height: Optional[int] = None
    scale_factor: Optional[float] = None
    draw_timestamp: bool = True
    font_path: Optional[Path] = None
    white_background: bool = True
