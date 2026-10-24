"""
Discrete RL state of Section 4.6.5:

    s_t = ( kappa_Theta(P_t), d_t, u_t, status_t, sigma_per(t) )

* kappa  - risk class of the LSTM probability (baixo | moderado | alto)
* d      - trend of recent probabilities (caindo | estavel | subindo)
* u      - uncertainty (baixa | alta); the custom LSTM has no Bayesian output,
           so u is approximated by the dispersion of the last predictions
* status - True if the context was evaluated (mandatory sources covered)
* period - situational period of the day

When the context is not evaluated, kappa/d/u are "na": no clinical finding
is produced from insufficient data (Property 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import pstdev
from typing import Sequence

from qvde_common.contexts import NOT_EVALUATED, risk_class, situational_period

NA = "na"


@dataclass(frozen=True)
class RLState:
    kappa: str
    trend: str
    uncertainty: str
    status: bool
    period: str

    @property
    def level(self) -> str:
        """Key of the action map alpha."""
        return self.kappa if self.status else NOT_EVALUATED

    def key(self) -> str:
        return f"{self.kappa}|{self.trend}|{self.uncertainty}|{int(self.status)}|{self.period}"

    def as_dict(self) -> dict:
        return {"classe": self.kappa, "tendencia": self.trend, "incerteza": self.uncertainty,
                "avaliado": self.status, "periodo": self.period}


def slope(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, sum(values) / n
    den = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (v - my) for i, v in enumerate(values)) / den


def trend_label(values: Sequence[float], delta: float) -> str:
    s = slope(values)
    if s > delta:
        return "subindo"
    if s < -delta:
        return "caindo"
    return "estavel"


def uncertainty_value(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def build_state(ctx: dict, p: float | None, recent: Sequence[float], available: bool,
                moment: datetime) -> tuple[RLState, dict]:
    """Return the discrete state and the numeric values behind it (for the uplink)."""
    period = situational_period(moment, ctx["contexto_situacional"]["periodo_do_dia"]) or NA
    if not available or p is None:
        return RLState(NA, NA, NA, False, period), {"p": None, "inclinacao": None, "u": None}
    lim = ctx["limiares"]
    series = list(recent) + [p]
    u = uncertainty_value(series[-5:])
    state = RLState(
        kappa=risk_class(p, lim["classe_risco"]),
        trend=trend_label(series[-5:], lim["tendencia_delta"]),
        uncertainty="alta" if u > lim["incerteza_alta"] else "baixa",
        status=True,
        period=period,
    )
    return state, {"p": round(p, 4), "inclinacao": round(slope(series[-5:]), 4), "u": round(u, 4)}
