from datetime import datetime, timezone

from qvde_common.contexts import situational_period
from qvde_common.timeutil import house_tz, local_time

RANGES = {"madrugada": [0, 6], "manha": [6, 12], "tarde": [12, 18], "noite": [18, 24]}


def test_period_uses_house_time_zone_not_utc():
    utc = datetime(2026, 11, 16, 18, 49, tzinfo=timezone.utc)          # 15:49 in Bahia
    assert situational_period(utc, RANGES) == "noite"                 # the old bug (container in UTC)
    assert situational_period(local_time(utc, house_tz("America/Bahia")), RANGES) == "tarde"
    assert situational_period(local_time(utc.timestamp(), house_tz("America/Bahia")), RANGES) == "tarde"


def test_madrugada_override_window_is_local():
    # 02:30 UTC = 23:30 in Bahia: must NOT be madrugada (the mandatory-notification override)
    t = datetime(2026, 11, 17, 2, 30, tzinfo=timezone.utc)
    assert situational_period(local_time(t, house_tz("America/Bahia")), RANGES) == "noite"


def test_default_and_env(monkeypatch):
    monkeypatch.delenv("QVDE_TIMEZONE", raising=False)
    assert str(house_tz()) == "America/Sao_Paulo"
    monkeypatch.setenv("QVDE_TIMEZONE", "America/Bahia")
    assert str(house_tz()) == "America/Bahia"
