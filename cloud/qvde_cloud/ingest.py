"""
Ingestion of the Hub uplink into the digital twin.

Each message is stored idempotently (a resent message is ignored) and then
acknowledged on qvde/v1/{house}/down/ack, which lets the Hub clean its
local buffer (Section 4.2.2). Messages whose topic and payload disagree, or
from unknown residences, are rejected and never acknowledged.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import Json

from qvde_common import protocol

log = logging.getLogger("qvde.ingest")


class Ingestor:
    def __init__(self, conn, reconnect=None):
        """reconnect: callable returning a new connection, used when the database restarts."""
        self.conn = conn
        self.reconnect = reconnect
        self._houses: set[str] = set()

    def _ensure_connection(self) -> None:
        if self.conn.closed and self.reconnect:
            log.warning("database connection closed; reconnecting")
            self.conn = self.reconnect()

    def known_house(self, house_id: str) -> bool:
        if house_id not in self._houses:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM residencias WHERE house_id=%s", (house_id,))
                if cur.fetchone():
                    self._houses.add(house_id)
        return house_id in self._houses

    def handle(self, topic: str, payload: bytes | str) -> str | None:
        """Store one message; return its msg_id to acknowledge, or None if rejected."""
        house, kind = protocol.parse_up_topic(topic)
        env = protocol.parse_envelope(payload, expected_house=house, expected_kind=kind)
        for attempt in (1, 2):
            self._ensure_connection()
            try:
                return self._store(house, kind, env)
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # database restarted: drop the broken connection and retry once
                if attempt == 2 or not self.reconnect:
                    raise
                try:
                    self.conn.close()
                except Exception:
                    pass
        return None

    def _store(self, house: str, kind: str, env: dict) -> str:
        if not self.known_house(house):
            raise ValueError(f"unknown residence {house}")
        handler = getattr(self, f"_store_{kind}")
        try:
            with self.conn, self.conn.cursor() as cur:
                handler(cur, env)
        except Exception:
            if not self.conn.closed:
                self.conn.rollback()
            raise
        return env["msg_id"]

    # ------------------------------------------------------------- per kind
    @staticmethod
    def _store_derivado(cur, env):
        d = env["dados"]
        cur.execute(
            """INSERT INTO derivados (ts, msg_id, house_id, sensor_id, contexto, versao_contexto,
                   avaliado, cobertura, p, inclinacao, u, classe, tendencia, incerteza, periodo,
                   acao, acoes_obrigatorias, modelo_versao, rl_versao, intervalo_s)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT DO NOTHING""",
            (env["ts"], env["msg_id"], env["house_id"], env.get("sensor_id"), d["contexto"],
             d.get("versao_contexto"), bool(d.get("avaliado")), d.get("cobertura"), d.get("p"),
             d.get("inclinacao"), d.get("u"), d.get("classe"), d.get("tendencia"),
             d.get("incerteza"), d.get("periodo"), d.get("acao"),
             list(d.get("acoes_obrigatorias", [])), d.get("modelo_versao"), d.get("rl_versao"),
             d.get("intervalo_s")))

    @staticmethod
    def _store_evento(cur, env):
        d = env["dados"]
        cur.execute(
            """INSERT INTO eventos (msg_id, ts, house_id, sensor_id, contexto, tipo_evento, p,
                   periodo, local, atividade_previa, modelo_versao)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            (env["msg_id"], env["ts"], env["house_id"], env.get("sensor_id"), d["contexto"],
             d["tipo_evento"], d.get("p"), d.get("periodo"), d.get("local"),
             d.get("atividade_previa"), d.get("modelo_versao")))

    @staticmethod
    def _store_episodio(cur, env):
        d = env["dados"]
        t_evento = datetime.fromtimestamp(d["t_evento"], tz=timezone.utc) if d.get("t_evento") else None
        cur.execute(
            """INSERT INTO episodios (msg_id, evento_id, house_id, sensor_id, contexto, rotulo,
                   rotulado_por, t_evento, campos, amostras)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            (env["msg_id"], d.get("evento_id"), env["house_id"], env.get("sensor_id"),
             d["contexto"], d.get("rotulo"), d.get("rotulado_por"), t_evento,
             list(d["campos"]), Json(d["amostras"])))

    @staticmethod
    def _store_estado(cur, env):
        cur.execute(
            """INSERT INTO estado_reportado (house_id, estado, atualizado) VALUES (%s,%s,now())
               ON CONFLICT (house_id) DO UPDATE SET estado=EXCLUDED.estado, atualizado=now()""",
            (env["house_id"], Json(env["dados"])))


def build_client(ingestor: Ingestor, host: str, port: int = 1883, user: str | None = None,
                 password: str | None = None, ca_file: str | None = None):
    """MQTT client that stores every uplink message and acknowledges it."""
    from qvde_common.mqtt_compat import make_client, rc_ok

    client = make_client("qvde-cloud-ingest")
    if ca_file:
        client.tls_set(ca_certs=ca_file, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    if user:
        client.username_pw_set(user, password)

    def on_connect(c, userdata, flags, reason_code, properties=None):
        if rc_ok(reason_code):
            c.subscribe(protocol.UP_WILDCARD, qos=1)
            log.info("subscribed to %s", protocol.UP_WILDCARD)

    def on_message(c, userdata, msg):
        try:
            msg_id = ingestor.handle(msg.topic, msg.payload)
        except Exception as exc:
            log.warning("rejected message on %s: %s", msg.topic, exc)
            return
        house, _ = protocol.parse_up_topic(msg.topic)
        c.publish(protocol.ack_topic(house), json.dumps({"ids": [msg_id]}), qos=1)

    client.on_connect, client.on_message = on_connect, on_message
    client.connect_async(host, port, keepalive=60)
    return client


def _connect_with_retry(connect, attempts: int = 30, delay_s: float = 2.0):
    """Wait for PostgreSQL (e.g. still initializing after a restart)."""
    import time
    for i in range(1, attempts + 1):
        try:
            return connect()
        except psycopg2.OperationalError as exc:
            if i == attempts:
                raise
            log.warning("database not ready (%s); retry %d/%d", str(exc).splitlines()[0], i, attempts)
            time.sleep(delay_s)


def main() -> None:  # pragma: no cover - network service
    from .db import apply_schema, connect

    logging.basicConfig(level=os.environ.get("QVDE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    conn = _connect_with_retry(connect)
    apply_schema(conn)
    client = build_client(Ingestor(conn, reconnect=lambda: _connect_with_retry(connect)),
                          os.environ.get("MQTT_HOST", "mosquitto"),
                          int(os.environ.get("MQTT_PORT", 1883)), os.environ.get("MQTT_USER"),
                          os.environ.get("MQTT_PASSWORD"), os.environ.get("MQTT_CA"))
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":  # pragma: no cover
    main()
