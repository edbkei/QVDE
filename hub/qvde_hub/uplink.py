"""
Connection from the Hub to the cloud digital twin.

* The Hub always initiates the connection (MQTT over TLS), so the residence
  exposes no inbound port.
* Messages are first written to the local outbox, then published; they stay
  pending until the cloud acknowledges them, so outages only delay delivery.
* The retained desired state updates the local shadow; the Hub then reports
  the versions it actually applied.
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from typing import Callable

from qvde_common import protocol

from .buffer import HubStore
from qvde_common.mqtt_compat import make_client, rc_ok
from .shadow import LocalShadow

log = logging.getLogger("qvde.uplink")


class CloudLink:
    def __init__(self, store: HubStore, shadow: LocalShadow, house_id: str,
                 host: str | None, port: int = 8883, ca_file: str | None = None,
                 cert_file: str | None = None, key_file: str | None = None,
                 username: str | None = None, password: str | None = None,
                 on_desired_applied: Callable[[dict], None] | None = None):
        self.store, self.shadow, self.house_id = store, shadow, house_id
        self.host, self.port = host, port
        self.on_desired_applied = on_desired_applied
        self.connected = threading.Event()
        self.client = None
        if not host:
            log.warning("CLOUD_BROKER not set: offline mode, messages stay queued locally")
            return
        self.client = make_client(f"qvde-hub-{house_id}")
        if ca_file or cert_file:
            self.client.tls_set(ca_certs=ca_file, certfile=cert_file, keyfile=key_file,
                                tls_version=ssl.PROTOCOL_TLS_CLIENT)
        if username:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.client:
            self.client.connect_async(self.host, self.port, keepalive=60)
            self.client.loop_start()

    def stop(self) -> None:
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    # ------------------------------------------------------------ callbacks
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if not rc_ok(reason_code):
            log.error("cloud connection refused: %s", reason_code)
            return
        client.subscribe([(protocol.ack_topic(self.house_id), 1),
                          (protocol.desired_topic(self.house_id), 1)])
        self.connected.set()
        log.info("connected to cloud %s:%s", self.host, self.port)
        self.report_state()

    def _on_disconnect(self, client, userdata, *args):
        self.connected.clear()
        log.warning("disconnected from cloud; queueing locally")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            if msg.topic == protocol.ack_topic(self.house_id):
                n = self.store.confirm([str(i) for i in data.get("ids", [])])
                log.debug("cloud confirmed %d messages", n)
            elif msg.topic == protocol.desired_topic(self.house_id):
                changed = self.shadow.apply_desired(data)
                if changed["contextos"] or changed["modelos"] or changed["erros"]:
                    log.info("desired state applied: %s", changed)
                    if self.on_desired_applied:
                        self.on_desired_applied(changed)
                self.report_state()
        except Exception:
            log.exception("bad downlink message on %s", msg.topic)

    # ---------------------------------------------------------------- sends
    def report_state(self) -> None:
        env = protocol.make_envelope("estado", self.house_id, self.shadow.reported())
        self.store.enqueue(env, protocol.up_topic(self.house_id, "estado"),
                           ts_ref=time.time())

    def flush(self, limit: int = 200) -> int:
        """Publish pending messages; returns how many were handed to the client."""
        if not self.client or not self.connected.is_set():
            return 0
        rows = self.store.pending(limit=limit)
        sent = []
        for msg_id, topic, payload in rows:
            info = self.client.publish(topic, payload, qos=1)
            if info.rc == 0:
                sent.append(msg_id)
        self.store.mark_sent(sent)
        return len(sent)
