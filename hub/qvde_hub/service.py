"""
QVDE Hub service: the reactive loop of Section 4.2.2.

    sensors --MQTT--> buffer --> context evaluator --> LSTM (P) --> state s_t
            --> RL (alpha, masked by the context) --> actuation (local)
            --> derived records / events / episodes --> outbox --> cloud

It runs next to the existing services and reads the same sensor topics
(iot/{house}/fall_detection/...), so it does not need InfluxDB.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from qvde_common import protocol
from qvde_common.contexts import (admissible_actions, coverage, hub_contexts, is_available)

from .buffer import HubStore
from .models import load_detection_model
from qvde_common.mqtt_compat import make_client, rc_ok
from .rl import ContextRLAgent, Decision
from .shadow import LocalShadow
from .state import build_state
from .uplink import CloudLink

log = logging.getLogger("qvde.hub")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MAP = {"accelerometer": "smartphone.imu", "imu": "smartphone.imu",
                      "smartphone": "smartphone.imu"}


@dataclass
class HubConfig:
    house_id: str = "house_001"
    local_broker: str = "mosquitto"
    local_port: int = 1883
    sensor_topic: str = "iot/+/fall_detection/+/+"
    action_topic: str = "qvde/local/{house}/acao"
    compat_prediction_topic: str = "ai/fall_prediction"
    db_path: str = "/app/data/hub.db"
    shadow_dir: str = "/app/data/shadow"
    contexts_dir: str = str(REPO / "contexts")
    model_dir: str = "/app/models"
    lstm_path: str | None = None
    rl_path: str | None = None
    interval_s: float = 3.0
    derived_every: int = 1
    action_repeat_s: float = 30.0
    cleanup_every_s: float = 600.0
    source_map: dict = field(default_factory=lambda: dict(DEFAULT_SOURCE_MAP))
    cloud_broker: str | None = None
    cloud_port: int = 8883
    cloud_ca: str | None = None
    cloud_cert: str | None = None
    cloud_key: str | None = None
    cloud_user: str | None = None
    cloud_password: str | None = None

    @classmethod
    def from_env(cls) -> "HubConfig":
        e = os.environ.get
        cfg = cls()
        for name, conv in (("house_id", str), ("local_broker", str), ("local_port", int),
                           ("sensor_topic", str), ("db_path", str), ("shadow_dir", str),
                           ("contexts_dir", str), ("model_dir", str), ("lstm_path", str),
                           ("rl_path", str), ("interval_s", float), ("derived_every", int),
                           ("cloud_broker", str), ("cloud_port", int), ("cloud_ca", str),
                           ("cloud_cert", str), ("cloud_key", str), ("cloud_user", str),
                           ("cloud_password", str)):
            value = e(f"QVDE_{name.upper()}")
            if value not in (None, ""):
                setattr(cfg, name, conv(value))
        if e("QVDE_SOURCE_MAP"):
            cfg.source_map.update(json.loads(e("QVDE_SOURCE_MAP")))
        return cfg


class HubService:
    def __init__(self, cfg: HubConfig):
        self.cfg = cfg
        Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.store = HubStore(cfg.db_path)
        self.shadow = LocalShadow(cfg.shadow_dir, bundled_contexts=cfg.contexts_dir)
        self._reload = threading.Event()
        self._stop = threading.Event()
        self.sources: dict[tuple[str, str], str] = {}       # (house, sensor) -> source id
        self.recent_p: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=10))
        self.last_action: dict[tuple, tuple[str, float]] = {}
        self.last_event: dict[tuple, float] = {}
        self.pending_episodes: list[dict] = []
        self.ticks = 0
        self._load_runtime()

        self.local = make_client(f"qvde-hub-local-{cfg.house_id}")
        self.local.on_connect = self._on_local_connect
        self.local.on_message = self._on_sensor
        self.cloud = CloudLink(self.store, self.shadow, cfg.house_id, cfg.cloud_broker,
                               cfg.cloud_port, cfg.cloud_ca, cfg.cloud_cert, cfg.cloud_key,
                               cfg.cloud_user, cfg.cloud_password,
                               on_desired_applied=lambda _changed: self._reload.set())

    # -------------------------------------------------------------- runtime
    def _load_runtime(self) -> None:
        self.contexts = hub_contexts(self.shadow.contexts(), self.shadow.active_contexts())
        self.model = load_detection_model(self.shadow.model_path("lstm") or self.cfg.lstm_path,
                                          self.cfg.model_dir)
        rl_path = self.shadow.model_path("rl") or self.cfg.rl_path
        self.agent = ContextRLAgent.load(rl_path) if rl_path and os.path.exists(rl_path) \
            else ContextRLAgent()
        log.info("hub contexts=%s model=%s rl=%s", sorted(self.contexts), self.model.version,
                 self.agent.version)

    def retention_s(self) -> float:
        days = [c.get("hub", {}).get("retencao_bruta_dias", 7) for c in self.contexts.values()]
        return max(days or [7]) * 86400.0

    # ---------------------------------------------------------------- input
    def _on_local_connect(self, client, userdata, flags, reason_code, properties=None):
        if rc_ok(reason_code):
            client.subscribe(self.cfg.sensor_topic, qos=1)
            log.info("subscribed to %s", self.cfg.sensor_topic)

    def _on_sensor(self, client, userdata, msg):
        try:
            self.ingest(json.loads(msg.payload), msg.topic)
        except Exception:
            log.exception("bad sensor message on %s", msg.topic)

    def ingest(self, payload: dict, topic: str = "", received: float | None = None) -> bool:
        """Store one sensor reading (existing simulator/sensor payload format)."""
        meta = payload.get("metadata", {})
        if meta.get("operational_mode") == "training" or "/training/" in topic:
            return False  # labelled training data is handled by the existing services
        house = meta.get("house_id")
        sensor = meta.get("sensor_id")
        if house != self.cfg.house_id or not sensor:
            return False
        received = received or time.time()
        ts = received
        if meta.get("timestamp"):
            try:
                t = datetime.fromisoformat(meta["timestamp"]).timestamp()
                ts = t if abs(t - received) < 300 else received  # guard against clock skew
            except ValueError:
                pass
        stype = payload.get("device_info", {}).get("sensor_type", "")
        self.sources[(house, sensor)] = self.cfg.source_map.get(stype, stype)
        self.store.add_sample(house, sensor, ts, payload.get("sensor_data", {}))
        return True

    # ----------------------------------------------------------------- tick
    def tick(self, now: float | None = None) -> list[dict]:
        """One cycle of the reactive loop. Returns the derived records produced."""
        now = now or time.time()
        if self._reload.is_set():
            self._reload.clear()
            self._load_runtime()
        self.ticks += 1
        records = []
        for (house, sensor), source in list(self.sources.items()):
            for cid, ctx in self.contexts.items():
                if source not in {f.get("sensor") for f in ctx.get("fontes", [])}:
                    continue
                records.append(self._evaluate(house, sensor, source, cid, ctx, now))
        self._finalize_episodes(now)
        return records

    def _evaluate(self, house, sensor, source, cid, ctx, now) -> dict:
        disp = ctx["disponibilidade"]
        rate = next(f.get("taxa_min_hz", 1) for f in ctx["fontes"] if f.get("sensor") == source)
        n = self.store.count_since(house, sensor, now - disp["janela_s"])
        cov = coverage(n, disp["janela_s"], rate)
        available = is_available({source: cov}, ctx)

        p = None
        key = (house, sensor, cid)
        if available:
            rows = self.store.recent(house, sensor, self.model.sequence_length)
            seq = np.array([r[1:] for r in rows], dtype=float)
            p = self.model.predict_proba(seq)

        moment = datetime.fromtimestamp(now)
        state, nums = build_state(ctx, p, self.recent_p[key], available, moment)
        if p is not None:
            self.recent_p[key].append(p)
        actions = admissible_actions(ctx, state.level, {"periodo_do_dia": state.period})
        decision = self.agent.decide(state, actions)
        self._actuate(house, sensor, cid, decision, state, nums, now)

        record = {"contexto": cid, "versao_contexto": ctx["versao"], "cobertura": round(cov, 3),
                  **state.as_dict(), **nums, "acao": decision.action,
                  "acoes_obrigatorias": list(decision.mandatory),
                  "modelo_versao": self.model.version, "rl_versao": self.agent.version,
                  "intervalo_s": self.cfg.interval_s}
        iso = datetime.fromtimestamp(now).astimezone().isoformat()
        if self.ticks % max(1, self.cfg.derived_every) == 0:
            env = protocol.make_envelope("derivado", house, record, sensor, ts=iso)
            self.store.enqueue(env, protocol.up_topic(house, "derivado"), ts_ref=now)

        lim = ctx["limiares"]
        if p is not None and (p >= lim.get("evento_queda", 0.9) or state.kappa == "alto"):
            self._register_event(house, sensor, cid, ctx, p, state, now, iso)
        return record

    # ------------------------------------------------------------ actuation
    def _actuate(self, house, sensor, cid, decision: Decision, state, nums, now) -> None:
        compat = {"fall_detected": (nums["p"] or 0) > 0.5, "fall_probability": nums["p"],
                  "fall_percentage": round((nums["p"] or 0) * 100, 2),
                  "rl_action": decision.action, "rl_state": state.key(),
                  "context": cid, "timestamp": datetime.now().isoformat(),
                  "model": "qvde_hub", "model_version": self.model.version}
        self.local.publish(self.cfg.compat_prediction_topic, json.dumps(compat), qos=0)
        for action in decision.all_actions:
            if action == "aguardar":
                continue
            last = self.last_action.get((house, sensor, action))
            if last and now - last[1] < self.cfg.action_repeat_s:
                continue
            self.last_action[(house, sensor, action)] = (action, now)
            self.local.publish(self.cfg.action_topic.format(house=house), json.dumps(
                {"acao": action, "contexto": cid, "sensor_id": sensor, "estado": state.as_dict(),
                 "p": nums["p"], "ts": datetime.fromtimestamp(now).astimezone().isoformat()}), qos=1)
            log.warning("action %s (context %s, state %s)", action, cid, state.key())

    # --------------------------------------------------- events and episodes
    def _register_event(self, house, sensor, cid, ctx, p, state, now, iso) -> None:
        window = ctx.get("hub", {}).get("janela_evento_s", 30)
        if now - self.last_event.get((house, sensor, cid), 0) < window:
            return
        self.last_event[(house, sensor, cid)] = now
        kind = "queda" if p >= ctx["limiares"].get("evento_queda", 0.9) else "quase_queda"
        data = {"contexto": cid, "tipo_evento": kind, "p": round(p, 4),
                "periodo": state.period, "modelo_versao": self.model.version}
        env = protocol.make_envelope("evento", house, data, sensor, ts=iso)
        self.store.enqueue(env, protocol.up_topic(house, "evento"), ts_ref=now)
        self.pending_episodes.append({"house": house, "sensor": sensor, "cid": cid, "t": now,
                                      "window": window, "evento_id": env["msg_id"], "tipo": kind})

    def _finalize_episodes(self, now: float) -> None:
        ready = [e for e in self.pending_episodes if now >= e["t"] + e["window"]]
        self.pending_episodes = [e for e in self.pending_episodes if e not in ready]
        for e in ready:
            rows = self.store.window(e["house"], e["sensor"], e["t"] - e["window"], e["t"] + e["window"])
            data = {"contexto": e["cid"], "evento_id": e["evento_id"], "rotulo": e["tipo"],
                    "rotulado_por": "hub", "t_evento": e["t"],
                    "campos": ["ts", "accel_x", "accel_y", "accel_z", "gyro_roll", "gyro_pitch", "gyro_yaw"],
                    "amostras": [[round(v, 4) for v in r] for r in rows]}
            env = protocol.make_envelope("episodio", e["house"], data, e["sensor"])
            self.store.enqueue(env, protocol.up_topic(e["house"], "episodio"), ts_ref=e["t"])

    # ----------------------------------------------------------------- main
    def run(self) -> None:
        self.local.connect_async(self.cfg.local_broker, self.cfg.local_port, keepalive=60)
        self.local.loop_start()
        self.cloud.start()
        last_cleanup = time.time()
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        try:
            while not self._stop.is_set():
                start = time.time()
                self.tick(start)
                self.cloud.flush()
                if start - last_cleanup > self.cfg.cleanup_every_s:
                    log.info("cleanup: %s", self.store.cleanup({self.cfg.house_id: self.retention_s()}))
                    last_cleanup = start
                self._stop.wait(max(0.0, self.cfg.interval_s - (time.time() - start)))
        except KeyboardInterrupt:
            pass
        finally:
            self.cloud.stop()
            self.local.loop_stop()
            self.local.disconnect()
            self.store.close()


def main() -> None:
    logging.basicConfig(level=os.environ.get("QVDE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    HubService(HubConfig.from_env()).run()


if __name__ == "__main__":
    main()
