import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from qvde_common.contexts import admissible_actions, load_context_dir
from qvde_hub.models import SMVHeuristic
from qvde_hub.rl import ContextRLAgent
from qvde_hub.service import HubConfig, HubService
from qvde_hub.shadow import LocalShadow
from qvde_hub.state import build_state, trend_label

ROOT = Path(__file__).resolve().parents[1]
CTX = load_context_dir(ROOT / "contexts")["risco_queda"]
NOON = datetime(2026, 10, 22, 12, 0)


def test_state_not_evaluated_produces_no_finding():
    s, nums = build_state(CTX, 0.95, [], available=False, moment=NOON)
    assert s.level == "nao_avaliado" and s.kappa == "na" and nums["p"] is None


def test_state_values():
    s, nums = build_state(CTX, 0.7, [0.3, 0.4, 0.5, 0.6], available=True, moment=NOON)
    assert (s.kappa, s.trend, s.period, s.status) == ("alto", "subindo", "tarde", True)
    assert trend_label([0.5, 0.5, 0.5], 0.05) == "estavel"
    assert nums["p"] == 0.7


def test_state_space_matches_thesis():
    kappas, trends, us, periods = 3, 3, 2, 4
    assert kappas * trends * us * 2 * periods == 144


def test_untrained_agent_follows_specialist_preference():
    agent = ContextRLAgent()
    s, _ = build_state(CTX, 0.8, [], True, NOON)
    d = agent.decide(s, admissible_actions(CTX, s.level, {"periodo_do_dia": s.period}))
    assert d.action == "alerta_local" and d.mandatory == ()


def test_night_override_adds_remote_notification():
    agent = ContextRLAgent()
    s, _ = build_state(CTX, 0.8, [], True, datetime(2026, 10, 22, 3, 0))
    d = agent.decide(s, admissible_actions(CTX, s.level, {"periodo_do_dia": s.period}))
    assert d.all_actions == ["alerta_local", "notificar_remoto"]


def test_agent_learns_and_masks(tmp_path):
    agent = ContextRLAgent(learning_rate=0.5)
    s, _ = build_state(CTX, 0.4, [], True, NOON)
    acts = admissible_actions(CTX, s.level, {"periodo_do_dia": s.period})
    for _ in range(20):  # moderate risk, real falls: waiting is punished
        agent.update(s, "aguardar", ContextRLAgent.reward("aguardar", True))
        agent.update(s, "notificar_remoto", ContextRLAgent.reward("notificar_remoto", True))
    assert agent.decide(s, acts).action == "notificar_remoto"
    agent.q[s.key()]["acionar_robo"] = 1e6     # not admissible at 'moderado'
    assert agent.decide(s, acts).action != "acionar_robo"
    agent.save(tmp_path / "rl.json")
    again = ContextRLAgent.load(tmp_path / "rl.json")
    assert again.q[s.key()] == agent.q[s.key()]


def test_smv_heuristic():
    m = SMVHeuristic()
    rest = np.tile([0, 0, 9.81, 0, 0, 0], (10, 1)).astype(float)
    fall = rest.copy(); fall[5, :3] = [20, 15, 5]
    assert m.predict_proba(rest) == 0.0 and m.predict_proba(fall) > 0.9


def test_shadow_applies_and_rejects(tmp_path):
    sh = LocalShadow(tmp_path / "sh", bundled_contexts=ROOT / "contexts")
    assert "risco_queda" in sh.contexts()
    text = (ROOT / "contexts" / "risco_queda.yaml").read_text().replace("versao: 1", "versao: 2", 1)
    model = tmp_path / "rl.json"; model.write_text("{}")
    changed = sh.apply_desired({"versao_estado": 1,
                                "contextos": {"risco_queda": {"versao": 2, "yaml": text},
                                              "ruim": {"versao": 1, "yaml": "id: [quebrado"}},
                                "modelos": {"rl": {"versao": "rl-1", "url": model.as_uri(),
                                                   "sha256": hashlib.sha256(b"{}").hexdigest()},
                                            "lstm": {"versao": "x", "url": model.as_uri(), "sha256": "0" * 64}},
                                "contextos_ativos": ["risco_queda"]})
    assert changed["contextos"] == ["risco_queda"] and changed["modelos"] == ["rl"]
    assert len(changed["erros"]) == 2                      # bad YAML and bad hash rejected
    assert LocalShadow(tmp_path / "sh").contexts()["risco_queda"]["versao"] == 2   # persisted
    assert sh.apply_desired({"versao_estado": 1})["contextos"] == []               # stale ignored


def sensor_msg(t: float, fall=False):
    acc = [25.0, 15.0, 5.0] if fall else [0.1, 0.0, 9.81]
    return {"metadata": {"house_id": "house_001", "sensor_id": "s1", "application": "fall_detection",
                         "timestamp": datetime.fromtimestamp(t).isoformat(), "operational_mode": "inference"},
            "device_info": {"sensor_type": "accelerometer", "location": "sala"},
            "sensor_data": dict(zip(["accel_x", "accel_y", "accel_z", "gyro_roll", "gyro_pitch", "gyro_yaw"],
                                    acc + [0, 0, 0]))}


def make_service(tmp_path):
    cfg = HubConfig(db_path=str(tmp_path / "hub.db"), shadow_dir=str(tmp_path / "sh"),
                    model_dir=str(tmp_path / "models"), contexts_dir=str(ROOT / "contexts"))
    return HubService(cfg)


def test_hub_tick_end_to_end(tmp_path):
    svc = make_service(tmp_path)
    t0 = datetime(2026, 10, 22, 14, 0).timestamp()
    # 10 s of normal walking at 10 Hz
    for i in range(100):
        svc.ingest(sensor_msg(t0 + i / 10), received=t0 + i / 10)
    rec = svc.tick(t0 + 10)[0]
    assert rec["avaliado"] and rec["classe"] == "baixo" and rec["acao"] == "aguardar"
    assert rec["modelo_versao"] == "heuristic-smv-v1"
    # a fall-like spike
    svc.ingest(sensor_msg(t0 + 10.05, fall=True), received=t0 + 10.05)
    rec = svc.tick(t0 + 10.1)[0]
    assert rec["classe"] == "alto" and rec["acao"] == "alerta_local"
    # 40 s later without data: context not evaluated, episode finalized
    rec = svc.tick(t0 + 50)[0]
    assert not rec["avaliado"] and rec["acao"] == "solicitar_dados"
    kinds = [json.loads(p)["tipo"] for _, _, p in svc.store.pending(limit=50)]
    assert kinds.count("derivado") == 3 and kinds.count("evento") == 1 and kinds.count("episodio") == 1
    episode = next(json.loads(p) for _, _, p in svc.store.pending(limit=50) if json.loads(p)["tipo"] == "episodio")
    assert episode["dados"]["rotulo"] == "queda" and len(episode["dados"]["amostras"]) > 50


def test_hub_ignores_training_and_other_houses(tmp_path):
    svc = make_service(tmp_path)
    m = sensor_msg(1e9)
    assert not svc.ingest(m, "iot/training/house_001/fall_detection/accelerometer/s1")
    m["metadata"]["house_id"] = "house_999"
    assert not svc.ingest(m)
