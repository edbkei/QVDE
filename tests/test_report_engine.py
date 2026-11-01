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


def test_pairing_is_one_to_one_and_composite_one_line_per_event():
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
    linhas = [a for a in achados if "após medida em pé" in a]
    assert len(linhas) == 3                                    # 13:54, 13:56, 13:59 (13:52 is before)
    assert "3 de 4 eventos ocorreram até 15 min após uma medida em pé." in achados
    assert achados[-1] == "Relação temporal (coocorrência), não causal."


def test_coverage_warning_uses_period_mean():
    from qvde_cloud.relatorio import proc_risco_queda
    entrada = {"probabilidade": 0.8, "horizonte_horas": None, "modelo": "proxy",
               "features": {"horas_uso_media_dia": 0.5, "dias_com_uso": "1 de 7"}, "eventos": [], "tendencia": []}
    achados = proc_risco_queda(entrada, _ctx("risco_queda"), {})["achados"]
    assert any("em média 0.5 h/dia" in a for a in achados)
