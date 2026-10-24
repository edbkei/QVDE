from datetime import datetime

import pytest

from qvde_common import contexts as c

CTX = c.load_context_dir(c.Path(__file__).resolve().parents[1] / "contexts")


def test_all_contexts_load_and_order():
    assert set(CTX) == {"risco_queda", "pressao_arterial", "hipotensao_ortostatica_queda"}
    order = c.topological_order(CTX)
    assert order.index("hipotensao_ortostatica_queda") > order.index("risco_queda")
    assert order.index("hipotensao_ortostatica_queda") > order.index("pressao_arterial")


def test_cycle_detected():
    bad = {"a": {"depende_de": ["b"]}, "b": {"depende_de": ["a"]}}
    with pytest.raises(c.ContextError):
        c.topological_order(bad)


def test_only_fall_risk_runs_on_hub():
    assert list(c.hub_contexts(CTX)) == ["risco_queda"]
    assert c.hub_contexts(CTX, active=["pressao_arterial"]) == {}


@pytest.mark.parametrize("n,expected", [(0, 0.0), (50, 0.5), (100, 1.0), (300, 1.0)])
def test_coverage(n, expected):
    assert c.coverage(n, window_s=10, rate_hz=10) == expected


def test_availability_uses_mandatory_sources_only():
    ctx = CTX["risco_queda"]
    assert c.is_available({"smartphone.imu": 0.6}, ctx)
    assert not c.is_available({"smartphone.imu": 0.59, "smartphone.barometro": 1.0}, ctx)


@pytest.mark.parametrize("p,level", [(0.0, "baixo"), (0.3, "moderado"), (0.59, "moderado"), (0.6, "alto")])
def test_risk_class(p, level):
    assert c.risk_class(p, CTX["risco_queda"]["limiares"]["classe_risco"]) == level


def test_period_and_overrides():
    ctx = CTX["risco_queda"]
    period = c.situational_period(datetime(2026, 1, 1, 3, 0), ctx["contexto_situacional"]["periodo_do_dia"])
    assert period == "madrugada"
    night = c.admissible_actions(ctx, "alto", {"periodo_do_dia": "madrugada"})
    day = c.admissible_actions(ctx, "alto", {"periodo_do_dia": "tarde"})
    assert night.mandatory == ("notificar_remoto",) and day.mandatory == ()
    assert c.admissible_actions(ctx, c.NOT_EVALUATED, {}).admissible[0] == "solicitar_dados"


def test_invalid_hub_context_rejected():
    with pytest.raises(c.ContextError):
        c.validate_context({"id": "x", "versao": 1, "nome": "x", "tipo": "simples", "execucao": "hub",
                            "disponibilidade": {}, "limiares": {}, "acoes": {"baixo": ["dancar"]}})
