"""
Tabular Q-learning agent over the discrete state of Section 4.6.5, with the
five actions a1..a5 and action masking by the care context's map alpha.

Ties (e.g. an untrained agent) are broken by the order of the admissible
actions in the YAML, so the default behaviour is the specialist's
preference, not a random choice. Exploration happens only in training.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from qvde_common.contexts import ACTIONS, ActionSet

from .state import RLState


@dataclass(frozen=True)
class Decision:
    action: str                 # chosen by the policy
    mandatory: tuple[str, ...]  # added by situational overrides

    @property
    def all_actions(self) -> list[str]:
        return [self.action] + [a for a in self.mandatory if a != self.action]


class ContextRLAgent:
    def __init__(self, learning_rate: float = 0.1, discount: float = 0.9,
                 epsilon: float = 0.0, version: str = "rl-ctx-v0"):
        self.q: dict[str, dict[str, float]] = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
        self.lr, self.gamma, self.epsilon = learning_rate, discount, epsilon
        self.version = version
        self.episodes = 0

    def decide(self, state: RLState, actions: ActionSet, explore: bool = False) -> Decision:
        options = list(actions.admissible) or ["aguardar"]
        if explore and random.random() < self.epsilon:
            choice = random.choice(options)
        else:
            qs = self.q[state.key()]
            best = max(qs[a] for a in options)
            choice = next(a for a in options if qs[a] == best)  # first = YAML preference
        return Decision(choice, actions.mandatory)

    def update(self, state: RLState, action: str, reward: float,
               next_state: RLState | None = None, next_actions: ActionSet | None = None) -> None:
        q_sa = self.q[state.key()][action]
        target = reward
        if next_state is not None:
            options = list(next_actions.admissible) if next_actions else list(ACTIONS)
            target += self.gamma * max(self.q[next_state.key()][a] for a in options)
        self.q[state.key()][action] = q_sa + self.lr * (target - q_sa)
        self.episodes += 1

    @staticmethod
    def reward(action: str, fall_occurred: bool, available: bool = True) -> float:
        """
        Reward used when training from labelled episodes. Missing a fall is the
        worst outcome; false alarms cost in proportion to how intrusive they are.
        """
        if not available:
            return 5.0 if action == "solicitar_dados" else -5.0
        if fall_occurred:
            return {"acionar_robo": 100.0, "alerta_local": 80.0, "notificar_remoto": 70.0,
                    "solicitar_dados": -50.0, "aguardar": -200.0}[action]
        return {"aguardar": 1.0, "solicitar_dados": -1.0, "alerta_local": -10.0,
                "notificar_remoto": -15.0, "acionar_robo": -30.0}[action]

    # JSON instead of pickle: the file may come from the cloud, and JSON cannot execute code.
    def save(self, path: str | Path) -> None:
        data = {"version": self.version, "lr": self.lr, "gamma": self.gamma,
                "epsilon": self.epsilon, "episodes": self.episodes, "q": dict(self.q)}
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ContextRLAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(data["lr"], data["gamma"], data["epsilon"], data["version"])
        agent.episodes = data["episodes"]
        for key, values in data["q"].items():
            agent.q[key] = {a: float(values.get(a, 0.0)) for a in ACTIONS}
        return agent
