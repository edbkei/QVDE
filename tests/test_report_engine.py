import pytest


def test_fidelity_metric():
    from qvde_cloud.relatorio import fidelidade
    f, sus = fidelidade("Quedas de 26/9 e 35/14 mmHg; p 0.68", {"a": "26/9 0.68"})
    assert sus == ["35/14"] and f == pytest.approx(2 / 3, abs=1e-3)
    assert fidelidade("nada numérico", {})[0] == 1.0
