import pytest


def test_fidelity_metric():
    from qvde_cloud.relatorio import fidelidade
    f, sus = fidelidade("Quedas de 26/9 e 35/14 mmHg; p 0.68", {"a": "26/9 0.68"})
    assert sus == ["35/14"] and f == pytest.approx(2 / 3, abs=1e-3)
    assert fidelidade("nada numérico", {})[0] == 1.0


def _ctx(cid):
    from pathlib import Path
    from qvde_cloud.relatorio import carregar_contexto
    return carregar_contexto(Path(__file__).resolve().parents[1] / "contexts", cid)


def test_pairing_is_one_to_one_and_composite_grouped_by_standing_measurement():
    """Two sitting measurements before one standing measurement form ONE pair (the closest)."""
    from qvde_cloud.relatorio import proc_ho_queda, proc_pressao_arterial
    pa = [
        {"timestamp": "2026-10-29T13:48:00+00:00", "sistolica": 148, "diastolica": 86, "posicao": "sentado"},
        {"timestamp": "2026-10-29T13:50:00+00:00", "sistolica": 152, "diastolica": 88, "posicao": "sentado"},
        {"timestamp": "2026-10-29T13:53:00+00:00", "sistolica": 128, "diastolica": 80, "posicao": "em_pe"},
    ]
    r_pa = proc_pressao_arterial(pa, _ctx("pressao_arterial"), {})
    pares = r_pa["dados"]["pares"]
    assert len(pares) == 1 and pares[0]["sentado"] == "152/88" and pares[0]["queda"] == "24/8"

    eventos = [{"timestamp": f"2026-10-29T13:{m}:00+00:00", "tipo": "quase_queda"} for m in (52, 54, 56, 59)]
    res = {"risco_queda": {"dados": {"eventos": eventos}}, "pressao_arterial": r_pa}
    achados = proc_ho_queda(None, _ctx("hipotensao_ortostatica_queda"), res)["achados"]
    linhas = [a for a in achados if "após a medida em pé" in a]
    # 13:54, 13:56, 13:59 share the same standing measurement -> one line (13:52 is before it)
    assert linhas == ["3 eventos 'quase_queda' em 2026-10-29 entre 13:54 e 13:59 ocorreram de 1 a 6 min "
                      "após a medida em pé de 2026-10-29 13:53 (queda de PA 24/8 mmHg; critério de "
                      "hipotensão ortostática atingido) [período: tarde]."]
    assert "3 de 4 eventos ocorreram até 15 min após uma medida em pé." in achados
    assert achados[-1] == "Relação temporal (coocorrência), não causal."


def test_coverage_warning_uses_period_mean():
    from qvde_cloud.relatorio import proc_risco_queda
    entrada = {"probabilidade": 0.8, "horizonte_horas": None, "modelo": "proxy",
               "features": {"horas_uso_media_dia": 0.5, "dias_com_uso": "1 de 7"}, "eventos": [], "tendencia": []}
    achados = proc_risco_queda(entrada, _ctx("risco_queda"), {})["achados"]
    assert any("em média 0.5 h/dia" in a for a in achados)


def test_events_grouped_and_no_fall_stated():
    from qvde_cloud.relatorio import proc_risco_queda
    eventos = [{"timestamp": f"2026-11-16T15:{m}:00-03:00", "tipo": "quase_queda", "periodo": "tarde"}
               for m in (49, 49, 50, 54)]
    entrada = {"probabilidade": 0.8, "modelo": "proxy", "features": {}, "eventos": eventos, "tendencia": []}
    achados = proc_risco_queda(entrada, _ctx("risco_queda"), {})["achados"]
    assert "4 eventos 'quase_queda' em 2026-11-16 entre 15:49 e 15:54 (período: tarde)." in achados
    assert "Nenhuma queda registrada: todos os 4 eventos são do tipo 'quase_queda' (sem queda)." in achados


def test_hub_period_wins_over_report_machine_clock():
    """The period comes from the Hub (house time zone), not from where the report runs."""
    from qvde_cloud.relatorio import descrever_situacao
    sit = _ctx("risco_queda")["contexto_situacional"]
    ev = {"timestamp": "2026-11-16T18:49:00+00:00", "tipo": "quase_queda", "periodo": "tarde"}
    assert descrever_situacao(ev, sit) == "período: tarde"
    ev_sem = {"timestamp": "2026-11-16T15:49:00-03:00", "tipo": "quase_queda"}
    assert descrever_situacao(ev_sem, sit) == "período: tarde"


def test_bp_trend_wording():
    from qvde_cloud.relatorio import proc_pressao_arterial
    def pa(vals):
        return [{"timestamp": f"2026-11-{10 + i:02d}T11:00:00-03:00", "sistolica": v, "diastolica": 86,
                 "posicao": "sentado"} for i, v in enumerate(vals)]
    estavel = proc_pressao_arterial(pa([148, 152, 150, 150]), _ctx("pressao_arterial"), {})["achados"][0]
    assert "estável em 150 mmHg" in estavel and "passou" not in estavel
    subiu = proc_pressao_arterial(pa([140, 142, 150, 152]), _ctx("pressao_arterial"), {})["achados"][0]
    assert "subiu de 141 para 151 mmHg" in subiu
