"""
Builds the digital-twin snapshot consumed by the report engine (the same JSON
format as exemplo_entrada.json). The snapshot carries the patient's
pseudonymous data, the fall-risk picture and the blood-pressure record.

If the daily fall-risk model (predicoes_risco) has no output yet, a clearly
labelled proxy is used: the daily 95th percentile of the Hub's detection
probability. It has no prediction horizon and the report says so.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

PROXY_MODEL = "agregado diário P95 da detecção no Hub (proxy; modelo de risco não implantado)"


def _iso(dt) -> str:
    return dt.astimezone().isoformat(timespec="minutes") if dt.tzinfo else dt.isoformat(timespec="minutes")


def build_snapshot(conn, patient_id: str, days: int = 7, bp_days: int = 30,
                   now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    since, bp_since = now - timedelta(days=days), now - timedelta(days=bp_days)
    with conn.cursor() as cur:
        cur.execute("""SELECT p.idade, p.sexo, p.condicoes, p.medicamentos, p.contextos_ativos, r.house_id
                       FROM pacientes p JOIN residencias r USING (paciente_id) WHERE p.paciente_id=%s""",
                    (patient_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"unknown patient {patient_id}")
        age, sex, conditions, meds, active, house = row
        snapshot = {"paciente": {"id": patient_id, "idade": age, "sexo": sex, "condicoes": list(conditions),
                                 "medicamentos": list(meds), "contextos_ativos": list(active)},
                    "gemeo_digital": {"origem": "QVDE cloud twin", "gerado_em": _iso(now)},
                    "prontuario": {}}

        fall = _fall_risk(cur, patient_id, house, since, now)
        if fall:
            snapshot["gemeo_digital"]["predicao_queda"] = fall

        cur.execute("""SELECT ts, sistolica, diastolica, fc, posicao, origem FROM observacoes_pa
                       WHERE paciente_id=%s AND ts>=%s ORDER BY ts""", (patient_id, bp_since))
        bp = [{"timestamp": _iso(ts), "sistolica": s, "diastolica": d, "fc": fc, "posicao": pos,
               "origem": orig} for ts, s, d, fc, pos, orig in cur.fetchall()]
        if bp:
            snapshot["prontuario"]["pressao_arterial"] = bp
    return snapshot


def _fall_risk(cur, patient_id, house, since, now) -> dict | None:
    cur.execute("""SELECT data, modelo, janela_h, horizonte_h, probabilidade, incerteza
                   FROM predicoes_risco WHERE paciente_id=%s AND data>=%s ORDER BY data""",
                (patient_id, since.date()))
    risk = cur.fetchall()
    cur.execute("""SELECT date_trunc('day', ts) AS dia,
                          percentile_cont(0.95) WITHIN GROUP (ORDER BY p) FILTER (WHERE avaliado) AS p95,
                          SUM(CASE WHEN avaliado THEN intervalo_s ELSE 0 END) / 3600.0 AS horas_uso
                   FROM derivados WHERE house_id=%s AND contexto='risco_queda' AND ts>=%s AND ts<%s
                   GROUP BY 1 ORDER BY 1""", (house, since, now))
    daily = cur.fetchall()
    cur.execute("""SELECT ts, tipo_evento, p, periodo, local, atividade_previa FROM eventos
                   WHERE house_id=%s AND contexto='risco_queda' AND ts>=%s ORDER BY ts""", (house, since))
    events = [{"timestamp": _iso(ts), "tipo": kind, "p": p, "periodo": per,
               **({"local": loc} if loc else {}), **({"atividade_previa": act} if act else {})}
              for ts, kind, p, per, loc, act in cur.fetchall()]
    if not risk and not any(p is not None for _, p, _ in daily):
        return None

    features = {}
    if daily:
        features["horas_uso_dispositivo"] = round(float(daily[-1][2] or 0), 1)
    if risk:
        last = risk[-1]
        return {"modelo": last[1], "janela_observacao_horas": last[2], "horizonte_horas": last[3],
                "probabilidade": round(last[4], 3), "incerteza": last[5],
                "tendencia": [{"data": d.isoformat(), "probabilidade": round(p, 3)} for d, _, _, _, p, _ in risk],
                "features": features, "eventos": events}
    series = [{"data": d.date().isoformat(), "probabilidade": round(float(p), 3)} for d, p, _ in daily if p is not None]
    return {"modelo": PROXY_MODEL, "janela_observacao_horas": 24, "horizonte_horas": None,
            "probabilidade": series[-1]["probabilidade"], "incerteza": None, "tendencia": series,
            "features": features, "eventos": events}
