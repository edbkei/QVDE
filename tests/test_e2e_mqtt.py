"""
End-to-end: sensor -> Hub -> (MQTT) -> cloud ingest -> PostgreSQL -> ack -> Hub cleanup,
and cloud desired state -> retained MQTT -> Hub shadow. Needs a mosquitto binary
on PATH and QVDE_TEST_DSN.
"""
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

DSN = os.environ.get("QVDE_TEST_DSN")
pytestmark = pytest.mark.skipif(not (DSN and shutil.which("mosquitto")),
                                reason="needs mosquitto and QVDE_TEST_DSN")
ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture()
def broker(tmp_path):
    port = free_port()
    conf = tmp_path / "m.conf"
    conf.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\n")
    proc = subprocess.Popen(["mosquitto", "-c", str(conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait(lambda: socket.socket().connect_ex(("127.0.0.1", port)) == 0)
    yield port
    proc.terminate()
    proc.wait()


def test_hub_to_cloud_roundtrip(broker, tmp_path, monkeypatch):
    from qvde_cloud.db import apply_schema, connect
    from qvde_cloud.ingest import Ingestor, build_client
    from qvde_cloud import twin_admin as T
    from qvde_hub.service import HubConfig, HubService
    from tests.test_hub_logic import sensor_msg

    conn = connect(DSN)
    with conn, conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    apply_schema(conn)
    T.register_residence(conn, "house_001", "PAC-0007", 81, "F", contexts=["risco_queda"])
    ingest = build_client(Ingestor(connect(DSN)), "127.0.0.1", broker)
    ingest.loop_start()

    cfg = HubConfig(db_path=str(tmp_path / "hub.db"), shadow_dir=str(tmp_path / "sh"),
                    model_dir=str(tmp_path / "m"), contexts_dir=str(ROOT / "contexts"),
                    local_broker="127.0.0.1", local_port=broker, cloud_broker="127.0.0.1",
                    cloud_port=broker, interval_s=0.2)
    hub = HubService(cfg)
    hub.cloud.start()
    assert hub.cloud.connected.wait(5)

    t0 = time.time() - 10
    for i in range(100):
        hub.ingest(sensor_msg(t0 + i / 10), received=t0 + i / 10)
    hub.ingest(sensor_msg(t0 + 9.95, fall=True), received=t0 + 9.95)
    hub.tick(t0 + 10)
    hub.tick(t0 + 50)          # finalizes the episode
    hub.cloud.flush()

    def count(table):
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            n = cur.fetchone()[0]
        conn.commit()
        return n

    assert wait(lambda: count("derivados") == 2 and count("eventos") == 1 and count("episodios") == 1)
    assert wait(lambda: hub.store.backlog() == 0)       # every message acknowledged
    # with everything confirmed, expired raw data can be removed on the Hub
    assert hub.store.cleanup({"house_001": 1.0}, now=t0 + 60)["raw"] > 0

    # desired state: publish new context version from the cloud, Hub applies it
    T.publish_contexts(conn, ROOT / "contexts")
    desired = T.build_desired(conn, "house_001")
    T.save_desired(conn, "house_001", desired)
    monkeypatch.setenv("MQTT_HOST", "127.0.0.1"); monkeypatch.setenv("MQTT_PORT", str(broker))
    T.publish_desired("house_001", desired)
    assert wait(lambda: hub.shadow.version == desired["versao_estado"])
    assert hub._reload.is_set()                          # service reloads contexts on next tick
    hub.cloud.flush()
    assert wait(lambda: T.status(conn, "house_001")["sincronizado"] or (hub.cloud.flush() and False), 10)

    hub.cloud.stop(); ingest.loop_stop(); ingest.disconnect()
