# src/dem_shadows/schedule.py
from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List
from astral import LocationInfo
from typing import Iterable, List, Tuple
from astral.sun import dawn as astral_dawn, dusk as astral_dusk
from zoneinfo import ZoneInfo
from .config import ShadowConfig

def _parse_hhmm(value: str) -> Tuple[int, int]:
    """
    Parse a 'HH:MM' string into (hour, minute).
    """
    hh, mm = value.split(":")
    return int(hh), int(mm)

def align_to_step(dt: datetime, minutes: int) -> datetime:
    """
    Align dt *forward* to the next multiple of `minutes` from midnight.

    Example:
      dt = 2025-03-08 05:21, minutes=60 -> 2025-03-08 06:00
      dt = 2025-03-08 05:21, minutes=15 -> 2025-03-08 05:30
    """
    from math import ceil

    minutes_since_midnight = dt.hour * 60 + dt.minute
    aligned_minutes = ceil(minutes_since_midnight / minutes) * minutes
    hour = aligned_minutes // 60
    minute = aligned_minutes % 60

    # If we rolled past midnight, clamp at 23:59 that day
    if hour >= 24:
        hour = 23
        minute = 59

    return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def iter_steps_aligned(start: datetime, end: datetime, minutes: int) -> Iterable[datetime]:
    """
    Yield timestamps from the first aligned step >= start, up to end (inclusive).
    """
    t = align_to_step(start, minutes)
    while t <= end:
        yield t
        t += timedelta(minutes=minutes)


def safe_dawn(location: LocationInfo, day: date, depression: float) -> datetime | None:
    """Return dawn or None if Astral raises (polar scenarios)."""
    try:
        return astral_dawn(location.observer, date=day, tzinfo=location.timezone, depression=depression)
    except ValueError:
        return None


def safe_dusk(location: LocationInfo, day: date, depression: float) -> datetime | None:
    """Return dusk or None if Astral raises (polar scenarios)."""
    try:
        return astral_dusk(location.observer, date=day, tzinfo=location.timezone, depression=depression)
    except ValueError:
        return None


def build_schedule(cfg: ShadowConfig) -> List[datetime]:
    """
    Two modes:
      - cfg.only_time set: one timestamp per day at that local time (if sun up)
      - otherwise: aligned intervals from dawn to dusk every step_minutes
    """
    # loc.timezone is a *string*; we also build a tzinfo object for datetimes
    loc = LocationInfo(
        name="site",
        region="",
        timezone=cfg.location.timezone,
        latitude=cfg.location.latitude,
        longitude=cfg.location.longitude,
    )
    local_tz = ZoneInfo(cfg.location.timezone)

    schedule: List[datetime] = []

    # --- Mode 1: one specific time per day, e.g. "10:15" ---
    if cfg.only_time is not None:
        hh, mm = _parse_hhmm(cfg.only_time)
        day = cfg.start_date
        while day <= cfg.end_date:
            # local time for that day with proper tzinfo object
            local_dt = datetime(day.year, day.month, day.day, hh, mm, tzinfo=local_tz)

            try:
                d = astral_dawn(
                    loc.observer,
                    date=day,
                    tzinfo=local_tz,                # use tzinfo object
                    depression=cfg.twilight_depression,
                )
                u = astral_dusk(
                    loc.observer,
                    date=day,
                    tzinfo=local_tz,
                    depression=cfg.twilight_depression,
                )
            except ValueError:
                day += timedelta(days=1)
                continue

            if d <= local_dt <= u:
                schedule.append(local_dt.astimezone(timezone.utc))

            day += timedelta(days=1)

        return sorted(set(schedule))

    # --- Mode 2: aligned intervals from dawn to dusk ---
    day = cfg.start_date
    while day <= cfg.end_date:
        try:
            d = astral_dawn(
                loc.observer,
                date=day,
                tzinfo=local_tz,                    # use tzinfo object
                depression=cfg.twilight_depression,
            )
            u = astral_dusk(
                loc.observer,
                date=day,
                tzinfo=local_tz,
                depression=cfg.twilight_depression,
            )
        except ValueError:
            day += timedelta(days=1)
            continue

        d_utc = d.astimezone(timezone.utc)
        u_utc = u.astimezone(timezone.utc)

        if u_utc < d_utc:
            end_today = datetime(day.year, day.month, day.day, 23, 59, tzinfo=timezone.utc)
            for t in iter_steps_aligned(d_utc, end_today, cfg.step_minutes):
                schedule.append(t)

            start_next = datetime(day.year, day.month, day.day, 0, 0, tzinfo=timezone.utc)
            for t in iter_steps_aligned(start_next, u_utc, cfg.step_minutes):
                schedule.append(t)
        else:
            for t in iter_steps_aligned(d_utc, u_utc, cfg.step_minutes):
                schedule.append(t)

        day += timedelta(days=1)

    return sorted(set(schedule))