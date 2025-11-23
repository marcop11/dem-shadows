from __future__ import annotations

from datetime import date, time as dtime
from pathlib import Path
import tempfile

import numpy as np
import rasterio
import streamlit as st

from dem_shadows.config import ShadowConfig, LocationConfig
from dem_shadows.shadows import run_shadow_batch
from dem_shadows.utils import get_dem_center_latlon, get_dem_timezone_tzfpy


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# Repo root (works when you run: streamlit run app/streamlit_app.py)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Example DEMs shipped with the repo.
EXAMPLE_DEMS = {
    "Zurich (dem.tif)": REPO_ROOT / "examples" / "dem.tif",
    "Iceland (dem_is.tif)": REPO_ROOT / "examples" / "dem_is.tif",
    "Paris (dem_fr.tif)": REPO_ROOT / "examples" / "dem_fr.tif",
}

# Logo (SVG) in img/
LOGO_PATH = REPO_ROOT / "img" / "logo.svg"

# Icon (SVG) in img/
ICON_PATH = REPO_ROOT / "img" / "icon.svg"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _load_for_visual(path: Path) -> np.ndarray:
    """Load a single-band raster and normalize for display (0..255)."""
    with rasterio.open(path) as src:
        arr = src.read(1)
    # If it's our shadow convention 0/1/255, make a nicer visualization:
    # 0 (shadow) -> dark, 1 (sunlit) -> bright, 255 (nodata) -> mid-gray
    unique_vals = np.unique(arr)
    if set(unique_vals.tolist()) <= {0, 1, 255}:
        vis = np.full(arr.shape, 128, dtype="uint8")  # nodata mid-gray
        vis[arr == 0] = 0
        vis[arr == 1] = 255
        return vis
    # Otherwise just rescale linearly
    arr = arr.astype("float32")
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr_n = (arr - amin) / (amax - amin)
    else:
        arr_n = np.zeros_like(arr)
    return (arr_n * 255).astype("uint8")


def _pick_dem_path(source: str, uploaded_file, example_label: str | None) -> Path | None:
    """
    Save uploaded DEM to a temp file or return example path.
    Returns a filesystem Path or None if nothing chosen.
    """
    if source == "Upload DEM GeoTIFF":
        if uploaded_file is None:
            return None
        tmp_dir = Path(tempfile.mkdtemp())
        dem_path = tmp_dir / "uploaded_dem.tif"
        dem_path.write_bytes(uploaded_file.getbuffer())
        return dem_path
    else:
        if example_label is None:
            return None
        path = EXAMPLE_DEMS.get(example_label)
        if path is None or not path.exists():
            st.error(
                f"Example DEM '{example_label}' not found. "
                "Check EXAMPLE_DEMS in streamlit_app.py."
            )
            return None
        return path

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="DEM Shadows",
    page_icon=str(ICON_PATH),   # use logo as tab icon
    layout="wide",
)

# Header with logo or title
if LOGO_PATH.exists():
    # Big logo as the “title”
    st.image(str(LOGO_PATH), width=360)
    st.caption("Generate terrain shadow rasters from DEM GeoTIFFs")
else:
    # Fallback: text title if logo missing
    st.title("🌄 DEM Shadows – Single Time Step")
    st.caption("Generate terrain shadow rasters from DEM GeoTIFFs")

st.markdown(
    """
This app takes a **DEM GeoTIFF** and generates a single **shadow raster**
for a chosen **date and local time**.

You can either:

1. Upload your own DEM, or  
2. Use one of the example DEMs bundled in this repo (`examples/*.tif`).
"""
)

# --- DEM source selection
st.sidebar.header("DEM source")
source = st.sidebar.radio(
    "Choose DEM",
    ["Upload DEM GeoTIFF", "Use example DEM from repo"],
)

uploaded = None
example_label = None

if source == "Upload DEM GeoTIFF":
    uploaded = st.sidebar.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])
else:
    example_label = st.sidebar.selectbox(
        "Select example DEM",
        list(EXAMPLE_DEMS.keys()),
    )

# --- Date and time
st.sidebar.header("Date & time")
sel_date = st.sidebar.date_input("Date", value=date(2025, 3, 8))
sel_time = st.sidebar.time_input("Local time", value=dtime(11, 55))

# --- Advanced options (optional)
st.sidebar.header("Advanced")
step_minutes = st.sidebar.number_input(
    "Step minutes (ignored when single time)",
    min_value=1,
    max_value=240,
    value=60,
)
twilight_dep = st.sidebar.number_input(
    "Twilight depression (deg)",
    min_value=0.0,
    max_value=18.0,
    value=6.0,
)

run_btn = st.sidebar.button("Generate shadow")


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
if run_btn:
    dem_path = _pick_dem_path(source, uploaded, example_label)
    if dem_path is None:
        st.error("Please provide a DEM or select a valid example DEM.")
        st.stop()

    # Show DEM quick info
    with rasterio.open(dem_path) as dem_src:
        dem_profile = dem_src.profile
    st.write("**DEM info**")
    st.json(
        {
            "path": str(dem_path),
            "width": dem_profile["width"],
            "height": dem_profile["height"],
            "crs": str(dem_profile["crs"]),
            "transform": str(dem_profile["transform"]),
        }
    )

    # Determine lat/lon and timezone automatically from the DEM
    try:
        lat, lon = get_dem_center_latlon(dem_path)
        tz_name = get_dem_timezone_tzfpy(dem_path)
    except Exception as e:
        st.error(f"Failed to derive location/timezone from DEM: {e}")
        st.stop()

    st.write(f"**Detected location**: lat={lat:.4f}, lon={lon:.4f}")
    st.write(f"**Detected timezone**: `{tz_name}`")

    # Build configuration: single time-of-day via only_time
    only_time_str = f"{sel_time.hour:02d}:{sel_time.minute:02d}"

    # Output folder in a temporary directory
    out_dir = Path(tempfile.mkdtemp())

    loc_cfg = LocationConfig(latitude=lat, longitude=lon, timezone=tz_name)
    cfg = ShadowConfig(
        dem_path=dem_path,
        out_dir=out_dir,
        location=loc_cfg,
        start_date=sel_date,
        end_date=sel_date,
        step_minutes=step_minutes,
        only_time=only_time_str,
        twilight_depression=twilight_dep,
        write_schedule_csv=False,
        overwrite=True,
    )

    st.write("Generating shadow… this may take a few seconds.")
    try:
        run_shadow_batch(cfg)
    except Exception as e:
        st.error(f"Shadow generation failed: {e}")
        st.stop()

    # Find resulting shadow file
    shadow_files = sorted(out_dir.glob("shadow_*.tif"))
    if not shadow_files:
        st.error("No shadow_*.tif generated. Check parameters and DEM.")
        st.stop()

    shadow_path = shadow_files[0]
    st.success(f"Shadow generated: {shadow_path.name}")

    # Display DEM & shadow side-by-side
    col_dem, col_shadow = st.columns(2)

    with col_dem:
        st.subheader("DEM (grayscale)")
        dem_vis = _load_for_visual(dem_path)
        st.image(dem_vis, clamp=True, use_container_width=True)

    with col_shadow:
        st.subheader("Shadow (0 = shadow, 1 = sunlit)")
        shadow_vis = _load_for_visual(shadow_path)
        st.image(shadow_vis, clamp=True, use_container_width=True)

    # Download link for the GeoTIFF
    with open(shadow_path, "rb") as f:
        shadow_bytes = f.read()

    st.download_button(
        label="⬇️ Download shadow GeoTIFF",
        data=shadow_bytes,
        file_name=shadow_path.name,
        mime="image/tiff",
    )

else:
    st.info("Select a DEM, choose a date & time, then click **Generate shadow** in the sidebar.")