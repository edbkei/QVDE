"""
Hub <-> cloud protocol (MQTT over TLS, always initiated by the Hub).

Uplink   qvde/v1/{house}/up/{kind}       kind in derivado | evento | episodio | estado
Downlink qvde/v1/{house}/down/ack        {"ids": [msg_id, ...]}
         qvde/v1/{house}/down/desejado   retained desired state (contexts and model versions)

Only derived data leaves the residence; raw IMU samples go up only inside
labelled event episodes (data minimization, Section 4.2.2).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

VERSION = "v1"
UP_KINDS = ("derivado", "evento", "episodio", "estado")


def up_topic(house_id: str, kind: str) -> str:
    if kind not in UP_KINDS:
        raise ValueError(f"unknown uplink kind {kind}")
    return f"qvde/{VERSION}/{house_id}/up/{kind}"


def ack_topic(house_id: str) -> str:
    return f"qvde/{VERSION}/{house_id}/down/ack"


def desired_topic(house_id: str) -> str:
    return f"qvde/{VERSION}/{house_id}/down/desejado"


UP_WILDCARD = f"qvde/{VERSION}/+/up/+"


def parse_up_topic(topic: str) -> tuple[str, str]:
    parts = topic.split("/")
    if len(parts) != 5 or parts[0] != "qvde" or parts[1] != VERSION or parts[3] != "up":
        raise ValueError(f"not an uplink topic: {topic}")
    if parts[4] not in UP_KINDS:
        raise ValueError(f"unknown uplink kind in {topic}")
    return parts[2], parts[4]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_envelope(kind: str, house_id: str, data: dict, sensor_id: str | None = None,
                  ts: str | None = None) -> dict:
    return {
        "msg_id": str(uuid.uuid4()),
        "tipo": kind,
        "house_id": house_id,
        "sensor_id": sensor_id,
        "ts": ts or now_iso(),
        "dados": data,
    }


def parse_envelope(raw: bytes | str, expected_house: str | None = None,
                   expected_kind: str | None = None) -> dict:
    env = json.loads(raw)
    for key in ("msg_id", "tipo", "house_id", "ts", "dados"):
        if key not in env:
            raise ValueError(f"envelope missing '{key}'")
    if env["tipo"] not in UP_KINDS:
        raise ValueError(f"unknown kind {env['tipo']}")
    if expected_house and env["house_id"] != expected_house:
        raise ValueError("house_id in payload does not match topic")
    if expected_kind and env["tipo"] != expected_kind:
        raise ValueError("tipo in payload does not match topic")
    datetime.fromisoformat(env["ts"])  # raises on malformed timestamps
    if not isinstance(env["dados"], dict):
        raise ValueError("dados must be an object")
    return env
