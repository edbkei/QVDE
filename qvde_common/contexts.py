"""
Care contexts (contextos de cuidado): loading, validation and the pure
functions of the formalization in Section 4.2.4 of the thesis.

  coverage c_s(W)          -> coverage()
  availability disp(C, t)  -> is_available()
  classification kappa     -> risk_class()
  situational sigma_per    -> situational_period()
  action map alpha         -> admissible_actions()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

#: RL actions of Section 4.6.5 (a1..a5)
ACTIONS = ("alerta_local", "notificar_remoto", "acionar_robo", "solicitar_dados", "aguardar")

NOT_EVALUATED = "nao_avaliado"
REQUIRED_KEYS = ("id", "versao", "nome", "tipo", "execucao")


class ContextError(ValueError):
    """Raised when a care context file is invalid."""


def validate_context(ctx: dict) -> dict:
    missing = [k for k in REQUIRED_KEYS if k not in ctx]
    if missing:
        raise ContextError(f"context {ctx.get('id', '?')}: missing keys {missing}")
    if ctx["tipo"] not in ("simples", "composto"):
        raise ContextError(f"context {ctx['id']}: tipo must be 'simples' or 'composto'")
    if ctx["execucao"] not in ("hub", "nuvem"):
        raise ContextError(f"context {ctx['id']}: execucao must be 'hub' or 'nuvem'")
    if ctx["tipo"] == "composto" and not ctx.get("depende_de"):
        raise ContextError(f"context {ctx['id']}: composite context needs depende_de")
    if ctx["execucao"] == "hub":
        for key in ("disponibilidade", "limiares", "acoes"):
            if key not in ctx:
                raise ContextError(f"context {ctx['id']}: hub context needs '{key}'")
        unknown = {a for lvl, acts in ctx["acoes"].items() if lvl != "sobreposicoes"
                   for a in acts} - set(ACTIONS)
        if unknown:
            raise ContextError(f"context {ctx['id']}: unknown actions {sorted(unknown)}")
    return ctx


def parse_context(text: str) -> dict:
    return validate_context(yaml.safe_load(text))


def load_context_dir(path: str | Path) -> dict[str, dict]:
    """Load every *.yaml in a directory (sub-directories such as perfis/ are ignored)."""
    contexts = {}
    for file in sorted(Path(path).glob("*.yaml")):
        ctx = parse_context(file.read_text(encoding="utf-8"))
        contexts[ctx["id"]] = ctx
    return contexts


def topological_order(contexts: dict[str, dict]) -> list[str]:
    """Evaluation order of the dependency DAG G (Property 2). Raises on cycles."""
    order, state = [], {}

    def visit(cid: str):
        if state.get(cid) == "done":
            return
        if state.get(cid) == "visiting":
            raise ContextError(f"dependency cycle at {cid}")
        state[cid] = "visiting"
        for dep in contexts.get(cid, {}).get("depende_de", []):
            if dep in contexts:
                visit(dep)
        state[cid] = "done"
        order.append(cid)

    for cid in contexts:
        visit(cid)
    return order


def coverage(n_samples: int, window_s: float, rate_hz: float) -> float:
    """c_s(W): fraction of the expected samples actually present in window W."""
    expected = window_s * rate_hz
    if expected <= 0:
        return 0.0
    return min(1.0, n_samples / expected)


def is_available(coverages: dict[str, float], ctx: dict) -> bool:
    """disp(C, t): every mandatory source reaches c_min (Definition 2)."""
    c_min = ctx["disponibilidade"]["cobertura_min"]
    mandatory = [f["sensor"] for f in ctx.get("fontes", []) if f.get("obrigatoria") and "sensor" in f]
    return all(coverages.get(s, 0.0) >= c_min for s in mandatory)


def risk_class(p: float, thresholds: dict) -> str:
    """kappa_Theta(p) with Theta = (moderado, alto)."""
    if p >= thresholds["alto"]:
        return "alto"
    if p >= thresholds["moderado"]:
        return "moderado"
    return "baixo"


def situational_period(moment: datetime, ranges: dict) -> str | None:
    """sigma_per(t) = k  <=>  hour(t) in I_k."""
    for name, (start, end) in ranges.items():
        if start <= moment.hour < end:
            return name
    return None


@dataclass(frozen=True)
class ActionSet:
    admissible: tuple[str, ...]   # the RL chooses one of these (YAML order = tie-break preference)
    mandatory: tuple[str, ...]    # always executed in addition (situational overrides)


def admissible_actions(ctx: dict, level: str, sigma: dict) -> ActionSet:
    """alpha: L x Sigma_C -> A, restricted by situational overrides."""
    acts = ctx["acoes"]
    admissible = tuple(acts.get(level, ("aguardar",)))
    mandatory: list[str] = []
    for rule in acts.get("sobreposicoes", []):
        cond = rule.get("quando", {})
        if cond.get("nivel", level) != level:
            continue
        if all(sigma.get(k) == v for k, v in cond.items() if k != "nivel"):
            mandatory.extend(a for a in rule.get("obrigatorias", []) if a not in mandatory)
    return ActionSet(admissible, tuple(mandatory))


def hub_contexts(contexts: dict[str, dict], active: Iterable[str] | None = None) -> dict[str, dict]:
    """Contexts evaluated in the reactive loop (C_ef^hub)."""
    act = set(active) if active is not None else None
    return {cid: c for cid, c in contexts.items()
            if c["execucao"] == "hub" and (act is None or cid in act)}
