# src/dem_shadows/animate.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
from typing import Iterable, List, Optional

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import re

from .config import AnimationConfig
from .utils import get_logger

logger = get_logger(__name__)


# Filename pattern: shadow_YYYYMMDDThhmmss[_TZ].tif
FNAME_RE = re.compile(
    r"shadow_(\d{8})T(\d{6})(?:_[^.]+)?\.tif$", re.IGNORECASE
)


def parse_timestamp_from_name(path: Path) -> Optional[datetime]:
    """
    Parse timestamp from a filename like:
        shadow_20250308T101500_EuropeZurich.tif
        shadow_20250308T101500.tif
    Returns a naive datetime (no timezone info attached).
    """
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    ymd, hms = m.groups()
    year = int(ymd[0:4])
    month = int(ymd[4:6])
    day = int(ymd[6:8])
    hour = int(hms[0:2])
    minute = int(hms[2:4])
    second = int(hms[4:6])
    return datetime(year, month, day, hour, minute, second)


def list_shadow_frames(
    folder: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    hour_filter: Optional[int] = None,
    minute_filter: Optional[int] = None,
) -> List[Path]:
    """
    Collect and sort shadow rasters by their timestamp parsed from filename.
    Applies optional date and time-of-day filters.
    """
    folder = Path(folder)
    candidates = sorted(folder.glob("shadow_*.tif"))
    rows = []

    for p in candidates:
        t = parse_timestamp_from_name(p)
        if t is None:
            continue

        d = t.date()
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue

        if hour_filter is not None and t.hour != hour_filter:
            continue
        if minute_filter is not None and t.minute != minute_filter:
            continue

        rows.append((t, p))

    rows.sort(key=lambda x: x[0])
    return [p for _, p in rows]



def resample_size(
    w: int,
    h: int,
    target_width: Optional[int],
    target_height: Optional[int],
    scale_factor: Optional[float],
) -> (int, int):
    """
    Compute resampled width/height based on either target size or scale factor.

    - If target_width or target_height is given, we scale to meet them
      while preserving aspect ratio.
    - Else if scale_factor is given, we scale by that factor.
    - Else returns (w, h).
    """
    if target_width or target_height:
        if target_width and not target_height:
            scale = target_width / w
        elif target_height and not target_width:
            scale = target_height / h
        else:
            scale = min(target_width / w, target_height / h)
    elif scale_factor:
        scale = scale_factor
    else:
        return (w, h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return (new_w, new_h)


def clean_clip_to_binary(
    arr: np.ndarray,
    nodata_value: int = 255,
) -> np.ndarray:
    """
    Enforce binary classification:

    - Values == nodata_value  -> kept as nodata_value
    - Other values are clipped to [0,1] and thresholded at 0.5:
        <0.5 -> 0 (shadow)
        >=0.5 -> 1 (sunlit)
    """
    arr = np.asarray(arr)
    out = np.full(arr.shape, nodata_value, dtype=np.uint8)

    valid = arr != nodata_value
    if not np.any(valid):
        return out

    vals = arr[valid].astype(np.float32)
    vals = np.clip(vals, 0.0, 1.0)
    out[valid] = (vals >= 0.5).astype(np.uint8)
    return out

def to_palette_image(
    arr_u8: np.ndarray,
    base_size: Optional[tuple[int, int]],
    cfg: AnimationConfig,
) -> Image.Image:
    """
    Convert uint8 array (0 shadow, 1 sun, 255 nodata) to a simple
    black/white image:

        - 0 (shadow)  -> 0   (black)
        - 1 (sunlit)  -> 255 (white)
        - 255 (nodata)-> 255 (white)
    """
    # Map to black/white
    bw = np.where(arr_u8 == 0, 0, 255).astype(np.uint8)

    # Grayscale image
    img = Image.fromarray(bw, mode="L")

    # Resize
    if base_size is None:
        new_size = resample_size(
            img.width,
            img.height,
            cfg.target_width,
            cfg.target_height,
            cfg.scale_factor,
        )
    else:
        new_size = base_size

    if new_size != (img.width, img.height):
        img = img.resize(new_size, resample=Image.NEAREST)

    return img


def add_label(
    imgP: Image.Image,
    text: str,
    cfg: AnimationConfig,
) -> Image.Image:
    """
    Draw timestamp text in the lower-left corner.

    Works on grayscale frames by temporarily converting to RGB.
    """
    if not cfg.draw_timestamp or not text:
        return imgP

    # Convert to RGB to draw colored text/box
    img = imgP.convert("RGB")
    d = ImageDraw.Draw(img)

    # Choose font
    try:
        if cfg.font_path:
            font = ImageFont.truetype(str(cfg.font_path), 16)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    pad = 6
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = pad
    y = img.height - th - pad

    # Black box
    d.rectangle(
        (x - 3, y - 2, x + tw + 3, y + th + 2),
        fill=(0, 0, 0),
    )
    # White text
    d.text((x, y), text, fill=(255, 255, 255), font=font)

    return img


def label_from_path(p: Path) -> str:
    t = parse_timestamp_from_name(p)
    if not t:
        return ""
    return t.strftime("%Y-%m-%d  %H:%M:%S")


def create_shadow_gif(cfg: AnimationConfig) -> None:
    """
    High-level: build ordered frame list, render palette frames, save GIF.
    """
    files_all = list_shadow_frames(
        folder=cfg.shadow_folder,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        hour_filter=cfg.hour_filter,
        minute_filter=cfg.minute_filter,
    )
    files = files_all[:: cfg.sample_stride]

    if not files:
        raise SystemExit(
            f"No shadow frames found in {cfg.shadow_folder} "
            f"for configured date/time filters."
        )

    logger.info("Animating %d frames (from %d candidates)", len(files), len(files_all))

    # First frame defines output size
    with rasterio.open(files[0]) as src0:
        arr0 = src0.read(1)
        nodata_val = src0.nodata if src0.nodata is not None else 255
        arr0 = clean_clip_to_binary(arr0, nodata_value=nodata_val)

        first_img = to_palette_image(
            arr0,
            base_size=None,
            cfg=cfg,
        )
        first_img = add_label(first_img, label_from_path(files[0]), cfg=cfg)

    frames: List[Image.Image] = []

    for fp in tqdm(files[1:], desc="Frames", unit="frame"):
        with rasterio.open(fp) as src:
            arr = src.read(1)
            nodata_val = src.nodata if src.nodata is not None else 255
            arr = clean_clip_to_binary(arr, nodata_value=nodata_val)

            im = to_palette_image(
                arr,
                base_size=first_img.size,
                cfg=cfg,
            )
            im = add_label(im, label_from_path(fp), cfg=cfg)
            frames.append(im)

    cfg.out_gif.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(
        save_all=True,
        append_images=frames,
        duration=cfg.duration_ms,
        loop=0,
        optimize=True,
    )

    first_img.save(cfg.out_gif, **save_kwargs)

    logger.info("Wrote GIF: %s (%d frames)", cfg.out_gif, 1 + len(frames))


def main_cli() -> None:
    """
    CLI entry point: dem-shadows-animate

    Example (auto for all frames in folder):

      dem-shadows-animate ^
        --shadow-folder "C:\\path\\to\\shadows" ^
        --out-gif "C:\\path\\to\\shadows.gif"

    Example (limit to one date and one local time-of-day):

      dem-shadows-animate ^
        --shadow-folder "C:\\path\\to\\shadows" ^
        --out-gif "C:\\path\\to\\shadows_10-15.gif" ^
        --start 2025-03-08 ^
        --end 2025-03-09 ^
        --hour 10 ^
        --minute 15
    """
    import argparse
    from datetime import datetime as dt

    parser = argparse.ArgumentParser(
        description="Create an animated GIF from shadow_*.tif rasters."
    )
    parser.add_argument(
        "--shadow-folder",
        type=Path,
        required=True,
        help="Folder containing shadow_*.tif rasters.",
    )
    parser.add_argument(
        "--out-gif",
        type=Path,
        required=True,
        help="Output GIF path.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD) inclusive.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) inclusive.",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help="Optional hour filter (0-23).",
    )
    parser.add_argument(
        "--minute",
        type=int,
        default=None,
        help="Optional minute filter (0-59).",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=250,
        help="Frame duration in milliseconds (default: 250).",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=900,
        help="Target width in pixels (default: 900, set 0 to disable).",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=0,
        help="Target height in pixels (default: 0 to auto / preserve aspect).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.0,
        help="Alternative to target size: scale factor (e.g. 0.5).",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamp label overlay.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to a TTF font file.",
    )

    args = parser.parse_args()

    start_date = dt.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = dt.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    cfg = AnimationConfig(
        shadow_folder=args.shadow_folder,
        out_gif=args.out_gif,
        start_date=start_date,
        end_date=end_date,
        hour_filter=args.hour,
        minute_filter=args.minute,
        duration_ms=args.duration_ms,
        sample_stride=args.sample_stride,
        target_width=args.target_width or None,
        target_height=args.target_height or None,
        scale_factor=args.scale_factor or None,
        draw_timestamp=not args.no_timestamp,
        font_path=args.font_path,
    )

    create_shadow_gif(cfg)