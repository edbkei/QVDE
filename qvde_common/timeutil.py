"""
House-local time. The situational context (period of the day) must be computed
in the time zone of the house, never in the time zone of the machine that runs
the code: the Hub container, the cloud VM and the laptop may all differ (Docker
containers default to UTC).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/Sao_Paulo"


def house_tz(name: str | None = None) -> ZoneInfo:
    """Time zone of the house: explicit name, else QVDE_TIMEZONE, else DEFAULT_TZ."""
    return ZoneInfo(name or os.environ.get("QVDE_TIMEZONE") or DEFAULT_TZ)


def local_time(moment, tz: ZoneInfo) -> datetime:
    """Epoch seconds or datetime -> aware datetime in tz (naive datetimes are taken as UTC)."""
    if isinstance(moment, (int, float)):
        return datetime.fromtimestamp(moment, tz)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(tz)
