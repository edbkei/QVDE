"""PostgreSQL helpers (psycopg2)."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
import psycopg2.extras

SCHEMA = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def connect(dsn: str | None = None):
    conn = psycopg2.connect(dsn or os.environ["DATABASE_URL"])
    psycopg2.extras.register_uuid()
    return conn


def apply_schema(conn) -> None:
    with conn, conn.cursor() as cur:
        cur.execute(SCHEMA.read_text(encoding="utf-8"))
