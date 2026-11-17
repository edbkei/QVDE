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

from qvde_common.timeutil import house_tz, local_time

PROXY_MODEL = "agregado diário P95 da detecção no Hub (proxy; modelo de risco não implantado)"


def _iso(dt, tz=None) -> str:
    """Timestamps go out in the house's time zone, whatever the machine running the report."""
    return local_time(dt, tz or house_tz()).isoformat(timespec="minutes")


def build_snapshot(conn, patient_id: str, days: int = 7, bp_days: int = 30,
                   now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    since, bp_since = now - timedelta(days=days), now - timedelta(days=bp_days)
    with conn.cursor() as cur:
        cur.execute("""SELECT p.idade, p.sexo, p.condicoes, p.medicamentos, p.contextos_ativos, r.house_id,
                              r.fuso_horario
                       FROM pacientes p JOIN residencias r USING (paciente_id) WHERE p.paciente_id=%s""",
                    (patient_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"unknown patient {patient_id}")
        age, sex, conditions, meds, active, house, tz_name = row
        tz = house_tz(tz_name)
        snapshot = {"paciente": {"id": patient_id, "idade": age, "sexo": sex, "condicoes": list(conditions),
                                 "medicamentos": list(meds), "contextos_ativos": list(active)},
                    "gemeo_digital": {"origem": "QVDE cloud twin", "gerado_em": _iso(now, tz),
                                      "fuso_horario": str(tz)},
                    "prontuario": {}}

        fall = _fall_risk(cur, patient_id, house, since, now, tz)
        if fall:
            snapshot["gemeo_digital"]["predicao_queda"] = fall

        cur.execute("""SELECT ts, sistolica, diastolica, fc, posicao, origem FROM observacoes_pa
                       WHERE paciente_id=%s AND ts>=%s ORDER BY ts""", (patient_id, bp_since))
        bp = [{"timestamp": _iso(ts, tz), "sistolica": s, "diastolica": d, "fc": fc, "posicao": pos,
               "origem": orig} for ts, s, d, fc, pos, orig in cur.fetchall()]
        if bp:
            snapshot["prontuario"]["pressao_arterial"] = bp
    return snapshot


def _fall_risk(cur, patient_id, house, since, now, tz=None) -> dict | None:
    tz = tz or house_tz()
    cur.execute("""SELECT data, modelo, janela_h, horizonte_h, probabilidade, incerteza
                   FROM predicoes_risco WHERE paciente_id=%s AND data>=%s ORDER BY data""",
                (patient_id, since.date()))
    risk = cur.fetchall()
    cur.execute("""SELECT date_trunc('day', ts AT TIME ZONE %s) AS dia,   -- house-local days
                          percentile_cont(0.95) WITHIN GROUP (ORDER BY p) FILTER (WHERE avaliado) AS p95,
                          SUM(CASE WHEN avaliado THEN intervalo_s ELSE 0 END) / 3600.0 AS horas_uso
                   FROM derivados WHERE house_id=%s AND contexto='risco_queda' AND ts>=%s AND ts<%s
                   GROUP BY 1 ORDER BY 1""", (str(tz), house, since, now))
    daily = cur.fetchall()
    cur.execute("""SELECT ts, tipo_evento, p, periodo, local, atividade_previa FROM eventos
                   WHERE house_id=%s AND contexto='risco_queda' AND ts>=%s ORDER BY ts""", (house, since))
    events = [{"timestamp": _iso(ts, tz), "tipo": kind, "p": p, "periodo": per,
               **({"local": loc} if loc else {}), **({"atividade_previa": act} if act else {})}
              for ts, kind, p, per, loc, act in cur.fetchall()]
    if not risk and not any(p is not None for _, p, _ in daily):
        return None

    # Device use over the whole period (not only its last day): mean hours/day,
    # number of days with use and the last day with use.
    features = {}
    period_days = (local_time(now, tz).date() - local_time(since, tz).date()).days + 1   # calendar days touched by the period
    used = [(d, float(h or 0)) for d, _, h in daily if (h or 0) > 0]
    features["horas_uso_media_dia"] = round(sum(h for _, h in used) / period_days, 2)
    features["dias_com_uso"] = f"{len(used)} de {period_days}"
    if used:
        features["ultimo_dia_com_uso"] = used[-1][0].date().isoformat()
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
