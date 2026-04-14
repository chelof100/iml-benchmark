"""
Drift Injector — controlled perturbations for 3 experimental scenarios.
All scenarios maintain: g(τ_t) = 0  ∀ t  (empirical instantiation of T2).
"""
import random
from dataclasses import dataclass
from typing import Dict

from iml.trace import Event, Trace
from iml.deviation import ALL_TOOLS, TOOL_RISK


@dataclass
class DriftConfig:
    mode: str           # "tool_drift" | "delegation_drift" | "context_drift"
    steps: int = 300
    burn_in: int = 50
    drift_start: int = 50   # when drift begins
    seed: int = 42


def _sample_tool(probs: Dict[str, float]) -> str:
    """
    Sample a tool given a probability dict over categories.
    probs keys: "safe", "boundary", "risky"
    All sampled tools are in ALL_TOOLS — no forbidden tools.
    """
    r = random.random()
    safe_tools = ["safe_read", "safe_query"]
    boundary_tools = ["moderate_write", "moderate_send"]
    risky_tools = ["risky_execute", "risky_delegate"]

    # normalize
    total = sum(probs.values())
    p_safe = probs.get("safe", 0.7) / total
    p_boundary = probs.get("boundary", 0.2) / total

    if r < p_safe:
        return random.choice(safe_tools)
    elif r < p_safe + p_boundary:
        return random.choice(boundary_tools)
    else:
        return random.choice(risky_tools)


def _make_event(tool: str, depth: int, agent: str = "A") -> Event:
    return Event(
        agent=agent,
        action="tool_call",
        tool=tool,
        depth=depth,
        metadata={"risk": TOOL_RISK.get(tool, 0.5)}
    )


class DriftInjector:
    """
    Generates a sequence of events with controlled drift.
    Key property: drift stays within enforcement constraints.
    Demonstrates that g(τ) = 0 while D̂(τ) grows.
    """

    def __init__(self, config: DriftConfig):
        self.config = config
        random.seed(config.seed)
        self.step = 0

        # Baseline probabilities (admission-time distribution)
        self.base_probs = {"safe": 0.75, "boundary": 0.20, "risky": 0.05}
        # Target after full drift
        self.target_probs = {
            "tool_drift":        {"safe": 0.15, "boundary": 0.75, "risky": 0.10},
            "delegation_drift":  {"safe": 0.70, "boundary": 0.20, "risky": 0.10},
            "context_drift":     {"safe": 0.20, "boundary": 0.60, "risky": 0.20},
        }

    def _alpha(self) -> float:
        """Linear interpolation factor: 0 → 1 over drift window."""
        drift_steps = self.config.steps - self.config.drift_start
        elapsed = max(0, self.step - self.config.drift_start)
        return min(1.0, elapsed / drift_steps)

    def _current_probs(self) -> Dict[str, float]:
        alpha = self._alpha()
        base = self.base_probs
        target = self.target_probs[self.config.mode]
        return {
            k: base[k] * (1 - alpha) + target[k] * alpha
            for k in base
        }

    def _current_depth(self) -> int:
        """
        For delegation_drift: depth increases gradually.
        For others: depth stays at 1.
        """
        if self.config.mode == "delegation_drift":
            alpha = self._alpha()
            return max(1, int(1 + alpha * 4))  # 1 → 5, never > MAX_ALLOWED_DEPTH
        return 1

    def next_event(self) -> Event:
        self.step += 1
        probs = self._current_probs()
        tool = _sample_tool(probs)
        depth = self._current_depth()
        return _make_event(tool, depth)

    def generate_burn_in(self, n: int) -> Trace:
        """Generate burn-in trace using base (admission-time) probabilities."""
        trace = Trace(trace_id="burn_in")
        random.seed(self.config.seed)
        for _ in range(n):
            tool = _sample_tool(self.base_probs)
            trace.add(_make_event(tool, depth=1))
        return trace
