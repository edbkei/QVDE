"""paho-mqtt 1.x / 2.x compatible client factory."""

from __future__ import annotations

import paho.mqtt.client as mqtt


def make_client(client_id: str) -> mqtt.Client:
    try:  # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:  # paho-mqtt 1.x
        return mqtt.Client(client_id=client_id)


def rc_ok(reason_code) -> bool:
    """reason_code is an int in paho 1.x and a ReasonCode in 2.x."""
    return getattr(reason_code, "value", reason_code) == 0
