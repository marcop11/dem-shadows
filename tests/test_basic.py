# tests/test_basic.py
from dem_shadows import __version__, ShadowConfig, LocationConfig

def test_imports():
    assert isinstance(__version__, str)
    loc = LocationConfig(latitude=0.0, longitude=0.0)
    ShadowConfig(
        dem_path="dem.tif",
        out_dir="out",
        location=loc,
        start_date=None,  # fill in real date in real tests
        end_date=None,
    )