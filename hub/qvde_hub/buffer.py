"""
Local storage of the Hub (SQLite, WAL mode).

* raw     - raw IMU samples, kept only for a configurable retention period
* outbox  - derived records waiting for (or confirmed by) the cloud
* meta    - small key/value store

Cleanup rule (Section 4.2.2): a raw sample is deleted only when it is older
than the retention period AND the cloud has confirmed every derived record up
to that time, so an outage longer than the retention never loses information.
Event windows are copied into an "episodio" message before they can expire.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

FIELDS = ("accel_x", "accel_y", "accel_z", "gyro_roll", "gyro_pitch", "gyro_yaw")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS raw (
    house_id TEXT NOT NULL,
    sensor_id TEXT NOT NULL,
    ts REAL NOT NULL,
    {", ".join(f"{f} REAL" for f in FIELDS)}
);
CREATE INDEX IF NOT EXISTS raw_by_sensor ON raw (house_id, sensor_id, ts);
CREATE INDEX IF NOT EXISTS raw_by_ts ON raw (ts);

CREATE TABLE IF NOT EXISTS outbox (
    msg_id TEXT PRIMARY KEY,
    house_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    topic TEXT NOT NULL,
    payload TEXT NOT NULL,
    ts_ref REAL NOT NULL,          -- epoch of the data the message describes
    created REAL NOT NULL,
    sent_at REAL,
    confirmed_at REAL
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox (confirmed_at, created);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class HubStore:
    def __init__(self, path: str | Path = ":memory:"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    # ------------------------------------------------------------------ raw
    def add_sample(self, house_id: str, sensor_id: str, ts: float, values: dict) -> None:
        row = [house_id, sensor_id, ts] + [float(values.get(f, 0.0)) for f in FIELDS]
        with self._lock:
            self._db.execute(f"INSERT INTO raw VALUES ({','.join('?' * len(row))})", row)

    def count_since(self, house_id: str, sensor_id: str, since: float) -> int:
        with self._lock:
            (n,) = self._db.execute(
                "SELECT COUNT(*) FROM raw WHERE house_id=? AND sensor_id=? AND ts>=?",
                (house_id, sensor_id, since)).fetchone()
        return n

    def recent(self, house_id: str, sensor_id: str, n: int) -> list[tuple]:
        """Last n samples as (ts, ax, ay, az, roll, pitch, yaw), oldest first."""
        with self._lock:
            rows = self._db.execute(
                f"SELECT ts, {', '.join(FIELDS)} FROM raw WHERE house_id=? AND sensor_id=? "
                "ORDER BY ts DESC LIMIT ?", (house_id, sensor_id, n)).fetchall()
        return rows[::-1]

    def window(self, house_id: str, sensor_id: str, t0: float, t1: float) -> list[tuple]:
        with self._lock:
            return self._db.execute(
                f"SELECT ts, {', '.join(FIELDS)} FROM raw WHERE house_id=? AND sensor_id=? "
                "AND ts BETWEEN ? AND ? ORDER BY ts", (house_id, sensor_id, t0, t1)).fetchall()

    def sensors_seen_since(self, since: float) -> list[tuple[str, str]]:
        with self._lock:
            return self._db.execute(
                "SELECT DISTINCT house_id, sensor_id FROM raw WHERE ts>=?", (since,)).fetchall()

    # --------------------------------------------------------------- outbox
    def enqueue(self, envelope: dict, topic: str, ts_ref: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO outbox (msg_id, house_id, kind, topic, payload, ts_ref, created) "
                "VALUES (?,?,?,?,?,?,?)",
                (envelope["msg_id"], envelope["house_id"], envelope["tipo"], topic,
                 json.dumps(envelope, ensure_ascii=False), ts_ref, time.time()))

    def pending(self, limit: int = 200, resend_after_s: float = 30.0) -> list[tuple[str, str, str]]:
        """Unconfirmed messages never sent, or sent long enough ago to be resent."""
        cutoff = time.time() - resend_after_s
        with self._lock:
            return self._db.execute(
                "SELECT msg_id, topic, payload FROM outbox WHERE confirmed_at IS NULL "
                "AND (sent_at IS NULL OR sent_at < ?) ORDER BY created LIMIT ?",
                (cutoff, limit)).fetchall()

    def mark_sent(self, msg_ids: list[str]) -> None:
        now = time.time()
        with self._lock:
            self._db.executemany("UPDATE outbox SET sent_at=? WHERE msg_id=?",
                                 [(now, m) for m in msg_ids])

    def confirm(self, msg_ids: list[str]) -> int:
        now = time.time()
        with self._lock:
            cur = self._db.executemany(
                "UPDATE outbox SET confirmed_at=? WHERE msg_id=? AND confirmed_at IS NULL",
                [(now, m) for m in msg_ids])
        return cur.rowcount

    def backlog(self) -> int:
        with self._lock:
            (n,) = self._db.execute("SELECT COUNT(*) FROM outbox WHERE confirmed_at IS NULL").fetchone()
        return n

    def confirmed_watermark(self, house_id: str) -> float:
        """Latest time up to which every derived record of the house is confirmed."""
        with self._lock:
            (oldest_open,) = self._db.execute(
                "SELECT MIN(ts_ref) FROM outbox WHERE house_id=? AND kind='derivado' "
                "AND confirmed_at IS NULL", (house_id,)).fetchone()
            bound = oldest_open if oldest_open is not None else float("inf")
            (last_conf,) = self._db.execute(
                "SELECT MAX(ts_ref) FROM outbox WHERE house_id=? AND kind='derivado' "
                "AND confirmed_at IS NOT NULL AND ts_ref<?", (house_id, bound)).fetchone()
        return last_conf if last_conf is not None else float("-inf")

    # -------------------------------------------------------------- cleanup
    def cleanup(self, retention_s: dict[str, float] | float, now: float | None = None,
                keep_confirmed_s: float = 86400.0) -> dict[str, int]:
        """
        Delete expired raw samples whose derived records are confirmed, and
        confirmed outbox rows older than keep_confirmed_s.

        retention_s: one value for every house, or {house_id: seconds}.
        """
        now = now or time.time()
        with self._lock:
            houses = [h for (h,) in self._db.execute("SELECT DISTINCT house_id FROM raw")]
        deleted_raw = 0
        for house in houses:
            ret = retention_s if isinstance(retention_s, (int, float)) else retention_s.get(house)
            if ret is None:
                continue
            limit = min(now - ret, self.confirmed_watermark(house))
            with self._lock:
                cur = self._db.execute("DELETE FROM raw WHERE house_id=? AND ts<?", (house, limit))
            deleted_raw += cur.rowcount
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM outbox WHERE confirmed_at IS NOT NULL AND confirmed_at<?",
                (now - keep_confirmed_s,))
        return {"raw": deleted_raw, "outbox": cur.rowcount}

    # ----------------------------------------------------------------- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def close(self) -> None:
        self._db.close()
