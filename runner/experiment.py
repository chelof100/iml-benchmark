"""
Experiment Runner
Runs all 3 drift scenarios, collects results, and saves to JSON.
This produces the data that validates T2 (g=0) and T3 (D̂ grows).
"""
import json
import os
import random
from typing import List, Dict

from iml.trace import Trace
from iml.deviation import IML, AdmissionSnapshot
from baselines.enforcement import enforcement_signal
from baselines.anomaly import AnomalyDetector
from runner.drift import DriftInjector, DriftConfig


def run_scenario(mode: str, steps: int = 300, seed: int = 42) -> List[Dict]:
    """
    Run one drift scenario.
    Returns list of per-step results with IML, enforcement, and anomaly signals.
    """
    config = DriftConfig(mode=mode, steps=steps, seed=seed)
    injector = DriftInjector(config)

    # Build A₀ from burn-in
    burn_in_trace = injector.generate_burn_in(config.burn_in)
    A0 = AdmissionSnapshot(burn_in_trace)

    # Initialize estimators
    iml = IML(A0)
    anomaly = AnomalyDetector(window_size=30)

    # Running trace (accumulates events)
    trace = Trace(trace_id=f"trace_{mode}")

    results = []

    for t in range(steps):
        event = injector.next_event()
        trace.add(event)

        # Compute signals
        D_hat = iml.compute(trace)
        breakdown = {
            "D_t": iml.D_t(trace),
            "D_c": iml.D_c(trace),
            "D_l": iml.D_l(trace),
        }
        g = enforcement_signal(trace)
        anomaly_score = anomaly.update(trace)

        results.append({
            "t": t,
            "mode": mode,
            "D_hat": round(D_hat, 4),
            "D_t": round(breakdown["D_t"], 4),
            "D_c": round(breakdown["D_c"], 4),
            "D_l": round(breakdown["D_l"], 4),
            "enforcement": g,
            "anomaly": round(anomaly_score, 4),
            "tool": event.tool,
            "depth": event.depth,
        })

    return results


def run_all(steps: int = 300, seed: int = 42, output_dir: str = "results") -> Dict:
    """
    Run all 3 scenarios and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for mode in ["tool_drift", "delegation_drift", "context_drift"]:
        print(f"Running scenario: {mode}...")
        results = run_scenario(mode=mode, steps=steps, seed=seed)
        all_results[mode] = results

        # Save per-scenario
        path = os.path.join(output_dir, f"{mode}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {path}")

        # Quick summary
        final_D = results[-1]["D_hat"]
        max_D = max(r["D_hat"] for r in results)
        max_g = max(r["enforcement"] for r in results)
        first_drift = next(
            (r["t"] for r in results if r["D_hat"] > 0.2), None
        )
        print(f"  D_hat final={final_D:.3f}  max={max_D:.3f}  "
              f"enforcement_max={max_g}  T*(0.20)={first_drift}")

    # Save summary
    summary = {
        mode: {
            "D_final": results[-1]["D_hat"],
            "D_max": max(r["D_hat"] for r in results),
            "D_at_t150": results[150]["D_hat"] if len(results) > 150 else None,
            "enforcement_triggers": sum(r["enforcement"] for r in results),
            "T_star_theta1": next(
                (r["t"] for r in results if r["D_hat"] > 0.20), None
            ),
            "T_star_theta2": next(
                (r["t"] for r in results if r["D_hat"] > 0.40), None
            ),
        }
        for mode, results in all_results.items()
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved -> {summary_path}")
    return summary
