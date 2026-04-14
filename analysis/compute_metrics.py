# -*- coding: utf-8 -*-
"""
Compute Summary Metrics from IML Experiment Results
====================================================
Paper: "From Admission to Invariants" (Fernandez, 2026)

Reads JSON result files and produces:
  - Console table matching paper Table 2 (300-step) and Table 3 (1000-step)
  - analysis/metrics_summary.json  -- machine-readable
  - analysis/results_table.txt     -- LaTeX-ready tabular rows

Usage:
  python analysis/compute_metrics.py
  python analysis/compute_metrics.py --dirs results results_1000 results_langgraph
  python analysis/compute_metrics.py --latex   # print LaTeX tabular rows
"""

import json
import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_run_metrics(results: List[Dict],
                        theta_1: float = 0.20,
                        theta_2: float = 0.35) -> Dict:
    """Compute all summary metrics for a single experiment run."""
    if not results:
        return {}

    n = len(results)
    D_values   = [r.get("D_hat",       0.0) for r in results]
    D_t_vals   = [r.get("D_t",         0.0) for r in results]
    D_c_vals   = [r.get("D_c",         0.0) for r in results]
    D_l_vals   = [r.get("D_l",         0.0) for r in results]
    enf_vals   = [r.get("enforcement", 0)   for r in results]
    t_vals     = [r.get("t",           i)   for i, r in enumerate(results)]

    D_final = D_values[-1]
    D_max   = max(D_values)
    D_init  = D_values[0]
    enf_total = sum(enf_vals)

    # T*(theta): first step where D_hat >= theta
    T_star_1 = next((t for t, d in zip(t_vals, D_values) if d >= theta_1), None)
    T_star_2 = next((t for t, d in zip(t_vals, D_values) if d >= theta_2), None)

    # Component breakdown at final step
    D_t_final = D_t_vals[-1]
    D_c_final = D_c_vals[-1]
    D_l_final = D_l_vals[-1]

    # Monotonicity: fraction of consecutive pairs where D_hat increases
    inc_pairs = sum(1 for i in range(1, n) if D_values[i] >= D_values[i-1])
    monotone_frac = inc_pairs / (n - 1) if n > 1 else 1.0

    # Drift rate: linear slope of D_hat over time
    if n > 1:
        t_arr = np.array(list(range(n)), dtype=float)
        d_arr = np.array(D_values)
        cov   = np.cov(t_arr, d_arr)
        drift_rate = float(cov[0, 1] / cov[0, 0]) if cov[0, 0] > 0 else 0.0
    else:
        drift_rate = 0.0

    # Anomaly baseline metrics (if present)
    anom_vals  = [r.get("anomaly",      None) for r in results]
    anom_vals  = [v for v in anom_vals if v is not None]
    anom_peak  = max(anom_vals) if anom_vals else None
    anom_final = anom_vals[-1]  if anom_vals else None
    anom_decay = (anom_peak - anom_final) if (anom_peak and anom_final) else None
    iml_lead   = (D_final - anom_final)   if anom_final is not None else None

    return {
        "n_steps":       n,
        "D_init":        round(D_init,        4),
        "D_final":       round(D_final,        4),
        "D_max":         round(D_max,          4),
        "D_t_final":     round(D_t_final,      4),
        "D_c_final":     round(D_c_final,      4),
        "D_l_final":     round(D_l_final,      4),
        "enforcement":   enf_total,
        "T_star_0.20":   T_star_1,
        "T_star_0.35":   T_star_2,
        "monotone_frac": round(monotone_frac,  4),
        "drift_rate":    round(drift_rate,      6),
        "anom_peak":     round(anom_peak,  4) if anom_peak  is not None else None,
        "anom_final":    round(anom_final, 4) if anom_final is not None else None,
        "anom_decay":    round(anom_decay, 4) if anom_decay is not None else None,
        "iml_lead":      round(iml_lead,   4) if iml_lead   is not None else None,
    }


# ─── Formatting ───────────────────────────────────────────────────────────────

def _fmt(val, fmt=".4f", none_str="---"):
    if val is None:
        return none_str
    return format(val, fmt)


def print_console_table(all_metrics: Dict[str, Dict]) -> None:
    """Print a human-readable summary table."""
    header = (
        f"{'Run':<35} {'n':>5} {'Enf':>4} {'D_final':>8} "
        f"{'T*(0.20)':>9} {'T*(0.35)':>9} {'D_t':>6} {'D_c':>6} {'D_l':>6}"
    )
    print("\n" + "=" * len(header))
    print("IML EXPERIMENT METRICS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for run_name, m in sorted(all_metrics.items()):
        if not m:
            continue
        t1 = str(m["T_star_0.20"]) if m["T_star_0.20"] is not None else "---"
        t2 = str(m["T_star_0.35"]) if m["T_star_0.35"] is not None else "---"
        print(
            f"{run_name:<35} {m['n_steps']:>5} {m['enforcement']:>4} "
            f"{m['D_final']:>8.4f} {t1:>9} {t2:>9} "
            f"{m['D_t_final']:>6.3f} {m['D_c_final']:>6.3f} {m['D_l_final']:>6.3f}"
        )

    print("=" * len(header) + "\n")


def print_latex_rows(all_metrics: Dict[str, Dict]) -> None:
    """Print LaTeX tabular rows for paper Table 2 / Table 3."""
    print("\n% LaTeX tabular rows (paste into paper):")
    print("% Scenario & Enf. & D_hat_final & T*_0.20 & Anom_peak & Anom_final"
          " & Anom_decay & IML_lead \\\\")
    for run_name, m in sorted(all_metrics.items()):
        if not m:
            continue
        short = run_name.split("/")[-1].replace("_drift", " drift").replace("_", " ").title()
        t1    = str(m["T_star_0.20"]) if m["T_star_0.20"] is not None else "---"
        ap    = _fmt(m["anom_peak"],  ".3f")
        af    = _fmt(m["anom_final"], ".3f")
        ad    = _fmt(m["anom_decay"], ".3f")
        il    = _fmt(m["iml_lead"],   ".3f")
        sign  = "+" if (m["iml_lead"] or 0) > 0 else ""
        il_s  = f"${sign}{il}$" if m["iml_lead"] is not None else "---"
        print(
            f"{short:<20} & \\textbf{{0}} & {m['D_final']:.3f} & {t1} "
            f"& {ap} & {af} & {ad} & {il_s} \\\\"
        )
    print()


# ─── Load results ─────────────────────────────────────────────────────────────

def load_results_dir(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all JSON files from a directory (skip summary.json)."""
    runs = {}
    if not os.path.isdir(results_dir):
        return runs
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json") or fname == "summary.json":
            continue
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                label = os.path.basename(results_dir) + "/" + fname.replace(".json", "")
                runs[label] = data
        except (json.JSONDecodeError, IOError):
            pass
    return runs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute IML experiment summary metrics"
    )
    parser.add_argument(
        "--dirs", nargs="+",
        default=["results", "results_1000", "results_langgraph"],
        help="Result directories to process"
    )
    parser.add_argument("--latex",  action="store_true", help="Print LaTeX rows")
    parser.add_argument("--out",    default="analysis/metrics_summary.json",
                        help="Output JSON path")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)

    # Collect all runs
    all_results: Dict[str, List[Dict]] = {}
    for d in args.dirs:
        full = d if os.path.isabs(d) else os.path.join(repo_root, d)
        all_results.update(load_results_dir(full))

    if not all_results:
        print("[error] No result files found. Run experiments first.")
        sys.exit(1)

    # Compute metrics for each run
    all_metrics: Dict[str, Dict] = {
        run: compute_run_metrics(data)
        for run, data in all_results.items()
    }

    # Output
    print_console_table(all_metrics)

    if args.latex:
        print_latex_rows(all_metrics)

    # Save JSON
    out_path = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
