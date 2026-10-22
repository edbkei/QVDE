"""
Cloud tests against a real PostgreSQL. Set QVDE_TEST_DSN to run them, e.g.
    QVDE_TEST_DSN=postgresql://postgres@localhost:5432/qvde_test pytest tests/test_cloud.py
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

DSN = os.environ.get("QVDE_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="QVDE_TEST_DSN not set")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def conn():
    from qvde_cloud.db import apply_schema, connect
    c = connect(DSN)
    with c, c.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    apply_schema(c)
    from qvde_cloud.twin_admin import register_residence
    register_residence(c, "house_001", "PAC-0007", 81, "F", ["hipertensão arterial"],
                       ["hidroclorotiazida 25 mg/dia"],
                       ["risco_queda", "pressao_arterial", "hipotensao_ortostatica_queda"])
    yield c
    c.close()


def envelope(kind, data, ts=None, house="house_001"):
    from qvde_common import protocol
    return protocol.up_topic(house, kind), protocol.make_envelope(kind, house, data, "s1", ts=ts)


DERIVED = {"contexto": "risco_queda", "versao_contexto": 1, "avaliado": True, "cobertura": 1.0,
           "classe": "baixo", "tendencia": "estavel", "incerteza": "baixa", "periodo": "tarde",
           "p": 0.1, "inclinacao": 0.0, "u": 0.0, "acao": "aguardar", "acoes_obrigatorias": [],
           "modelo_versao": "heuristic-smv-v1", "rl_versao": "rl-ctx-v0", "intervalo_s": 3.0}


def test_ingest_is_idempotent_and_validates(conn):
    from qvde_cloud.ingest import Ingestor
    ing = Ingestor(conn)
    topic, env = envelope("derivado", DERIVED)
    assert ing.handle(topic, json.dumps(env)) == env["msg_id"]
    assert ing.handle(topic, json.dumps(env)) == env["msg_id"]        # resend: acked, stored once
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM derivados"); assert cur.fetchone()[0] == 1
    with pytest.raises(ValueError):                                    # payload house != topic house
        ing.handle(topic.replace("house_001", "house_002"), json.dumps(env))
    t2, e2 = envelope("derivado", DERIVED, house="house_999")
    with pytest.raises(ValueError):                                    # unknown residence
        ing.handle(t2, json.dumps(e2))


def test_ingest_event_episode_state(conn):
    from qvde_cloud.ingest import Ingestor
    ing = Ingestor(conn)
    t, e = envelope("evento", {"contexto": "risco_queda", "tipo_evento": "quase_queda", "p": 0.7, "periodo": "manha"})
    ing.handle(t, json.dumps(e))
    t, ep = envelope("episodio", {"contexto": "risco_queda", "evento_id": e["msg_id"], "rotulo": "quase_queda",
                                  "rotulado_por": "hub", "t_evento": 1.7e9, "campos": ["ts", "accel_x"],
                                  "amostras": [[1.7e9, 0.1]]})
    ing.handle(t, json.dumps(ep))
    t, st = envelope("estado", {"versao_estado": 3, "contextos": {"risco_queda": 1}, "modelos": {}})
    ing.handle(t, json.dumps(st))
    with conn.cursor() as cur:
        cur.execute("SELECT rotulo, amostras FROM episodios"); row = cur.fetchone()
        assert row[0] == "quase_queda" and row[1] == [[1.7e9, 0.1]]
        cur.execute("SELECT estado->>'versao_estado' FROM estado_reportado"); assert cur.fetchone()[0] == "3"


def test_contexts_versions_and_desired_state(conn, tmp_path):
    from qvde_cloud import twin_admin as T
    assert len(T.publish_contexts(conn, ROOT / "contexts")) == 3
    assert T.publish_contexts(conn, ROOT / "contexts") == []           # same versions: nothing new
    # editing a published version without bumping it is refused
    d = tmp_path / "ctx"; d.mkdir()
    for f in (ROOT / "contexts").glob("*.yaml"):
        (d / f.name).write_text(f.read_text())
    (d / "risco_queda.yaml").write_text((d / "risco_queda.yaml").read_text().replace("cobertura_min: 0.6", "cobertura_min: 0.7"))
    with pytest.raises(SystemExit):
        T.publish_contexts(conn, d)
    model = tmp_path / "rl.json"; model.write_text("{}")
    T.register_model(conn, "rl", "rl-1", "https://example.org/rl.json", file=model)
    desired = T.build_desired(conn, "house_001", {"rl": "rl-1"})
    assert desired["versao_estado"] == 1 and set(desired["contextos"]) == {
        "risco_queda", "pressao_arterial", "hipotensao_ortostatica_queda"}
    assert desired["modelos"]["rl"]["sha256"]
    T.save_desired(conn, "house_001", desired)
    assert T.build_desired(conn, "house_001")["versao_estado"] == 2     # keeps chosen models, bumps version
    assert T.build_desired(conn, "house_001")["modelos"]["rl"]["versao"] == "rl-1"
    assert T.status(conn, "house_001")["sincronizado"] is False


def seed_week(conn, now):
    """7 days of derived records with rising detection, 2 near-falls after orthostatic drops."""
    from qvde_cloud.ingest import Ingestor
    from qvde_cloud.twin_admin import import_bp
    ing = Ingestor(conn)
    for day in range(7):
        for k in range(20):
            ts = (now - timedelta(days=6 - day, hours=k % 10)).isoformat()
            d = dict(DERIVED, p=round(0.05 * day + 0.01 * k, 3))
            t, e = envelope("derivado", d, ts=ts); ing.handle(t, json.dumps(e))
    base = now.replace(hour=11, minute=0, second=0, microsecond=0)
    rows = ["timestamp,sistolica,diastolica,fc,posicao,origem"]
    for day, (sit, up) in enumerate([((148, 86), (126, 80)), ((152, 88), (135, 82)), ((156, 90), (130, 81))]):
        t0 = base - timedelta(days=4 - 2 * day)
        rows.append(f"{t0.isoformat()},{sit[0]},{sit[1]},72,sentado,MRPA")
        rows.append(f"{(t0 + timedelta(minutes=3)).isoformat()},{up[0]},{up[1]},86,em_pe,MRPA")
        if day != 1:
            t, e = envelope("evento", {"contexto": "risco_queda", "tipo_evento": "quase_queda", "p": 0.72,
                                       "periodo": "manha"}, ts=(t0 + timedelta(minutes=9)).isoformat())
            ing.handle(t, json.dumps(e))
    csv_path = Path(os.environ.get("TMPDIR", "/tmp")) / "pa_test.csv"
    csv_path.write_text("\n".join(rows))
    assert import_bp(conn, "PAC-0007", csv_path) == 6


def test_snapshot_and_report_without_llm(conn):
    from qvde_cloud.report_job import generate, store
    from qvde_cloud.snapshot import PROXY_MODEL, build_snapshot
    now = datetime.now(timezone.utc)
    seed_week(conn, now)
    snap = build_snapshot(conn, "PAC-0007", now=now + timedelta(minutes=1))
    fall = snap["gemeo_digital"]["predicao_queda"]
    assert fall["modelo"] == PROXY_MODEL and fall["horizonte_horas"] is None
    assert len(fall["tendencia"]) >= 7 and len(fall["eventos"]) == 2
    assert len(snap["prontuario"]["pressao_arterial"]) == 6

    out = generate(conn, "PAC-0007", str(ROOT / "contexts/perfis/geriatra.yaml"), call_llm=False)
    md = out["markdown"]
    assert "sem horizonte de predição" in md                 # proxy is declared, not disguised
    assert "ATINGE critério de hipotensão ortostática" in md
    assert md.count("após medida em pé") == 2                # composite context found both pairs
    assert "Hipotensão ortostática e quedas" in md
    rid = store(conn, "PAC-0007", out)
    assert rid > 0
