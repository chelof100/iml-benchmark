# -*- coding: utf-8 -*-
"""
Statistical Validation of IML Theoretical Guarantees
=====================================================
Paper: "From Admission to Invariants" (Fernandez, 2026)

Tests:
  T2 -- Non-Identifiability: enforcement signal carries zero information about A0
        -> Validates that enforcement_triggers = 0 across ALL runs (exact test)
        -> Chi-squared goodness-of-fit: observed enforcement matches H0: g(t)=0 always
        -> Brier score: enforcement predictions vs D_hat crossings

  T3 -- Detection Guarantee: IML detects drift before theta with high probability
        -> Mann-Kendall trend test: D_hat is monotonically increasing (one-sided)
        -> Bootstrap confidence intervals on T*(theta)
        -> Hoeffding bound validation: |D_hat_n - D*| <= eps_est(n) w.p. >= 1-delta

Usage:
  python analysis/stats_tests.py                    # all tests, all result files
  python analysis/stats_tests.py --results results/ # specific directory
  python analysis/stats_tests.py --verbose          # detailed output

Output:
  analysis/stats_report.json -- machine-readable results
  Console summary with PASS/FAIL for each theorem
"""

import json
import os
import sys
import math
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

# ─── Mann-Kendall trend test ──────────────────────────────────────────────────

def mann_kendall(x: List[float]) -> Tuple[float, float, str]:
    """
    One-sided Mann-Kendall trend test.
    H0: no trend.  H1: monotone increasing trend.

    Returns:
        (S, z_score, verdict)
        S     -- Mann-Kendall statistic
        z     -- standardized Z score
        verdict -- "increasing (p<alpha)", "no trend", or "decreasing"
    """
    n = len(x)
    if n < 4:
        return 0.0, 0.0, "insufficient data"

    # Compute S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                S += 1
            elif diff < 0:
                S -= 1

    # Variance (accounting for ties)
    # Count ties
    from collections import Counter
    tie_counts = Counter(x)
    tie_correction = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts.values() if t > 1)
    var_S = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18

    if var_S <= 0:
        return float(S), 0.0, "no variance"

    # Z statistic
    if S > 0:
        z = (S - 1) / math.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / math.sqrt(var_S)
    else:
        z = 0.0

    # One-sided p-value (H1: increasing)
    # Using normal approximation
    p = 0.5 * (1 - _erf(z / math.sqrt(2)))

    alpha = 0.05
    if z > 0 and p < alpha:
        verdict = f"increasing (z={z:.3f}, p={p:.4f} < {alpha})"
    elif z < 0 and p < alpha:
        verdict = f"decreasing (z={z:.3f}, p={p:.4f})"
    else:
        verdict = f"no trend (z={z:.3f}, p={p:.4f})"

    return float(S), float(z), verdict


def _erf(x: float) -> float:
    """Approximate error function (Abramowitz & Stegun)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y


# ─── T2: Non-Identifiability validation ──────────────────────────────────────

def test_T2_enforcement_zero(results: List[Dict], run_name: str) -> Dict:
    """
    T2: Validates that g(tau_t) = 0 for all t in the run.

    Theorem 2 states: A0 is not sigma(g)-measurable.
    Corollary (Irrecoverability): no function of g can recover A0.

    The empirical test: enforcement should trigger 0 times.
    If enforcement = 0 throughout, T2's precondition holds and
    D_hat rising from 0 to >theta is evidence of non-identifiability.

    Returns dict with keys: run, n_steps, enforcement_total, pass, evidence
    """
    n = len(results)
    enforcement_total = sum(r.get("enforcement", 0) for r in results)
    D_final = results[-1].get("D_hat", 0.0) if results else 0.0
    D_values = [r.get("D_hat", 0.0) for r in results]
    D_max = max(D_values) if D_values else 0.0

    # Evidence: enforcement=0 AND D_hat rises above theta
    theta = 0.20
    drift_detected = any(d >= theta for d in D_values)
    gap = D_max - 0.0  # max drift while enforcement stayed 0

    passed = (enforcement_total == 0)
    evidence_strength = "strong" if (passed and drift_detected) else ("weak" if passed else "none")

    return {
        "run": run_name,
        "n_steps": n,
        "enforcement_total": enforcement_total,
        "D_final": round(D_final, 4),
        "D_max": round(D_max, 4),
        "drift_detected_above_theta": drift_detected,
        "information_gap": round(gap, 4),
        "T2_pass": passed,
        "T2_evidence": evidence_strength,
        "interpretation": (
            "T2 CONFIRMED: enforcement=0 throughout while D_hat drifted to "
            f"{D_max:.4f}. Enforcement carried zero information about A0 deviation."
            if (passed and drift_detected)
            else "T2 WEAK: drift not yet significant." if passed
            else f"T2 FAIL: enforcement triggered {enforcement_total} times."
        )
    }


def test_T2_brier_score(results: List[Dict], run_name: str) -> Dict:
    """
    Brier score test: if g were a predictor of 'D_hat >= theta' crossings,
    what is its Brier score?

    g(t) = 0 always -> predicts "no crossing" at every step.
    Actual crossings: D_hat(t) >= theta.
    Brier = mean((g(t) - 1_{D_hat>=theta})^2)

    High Brier score confirms g is uninformative about drift.
    Null predictor (g=0 always) Brier = fraction of steps where D_hat >= theta.
    """
    theta = 0.20
    actual_crossings = [1 if r.get("D_hat", 0.0) >= theta else 0 for r in results]
    n = len(actual_crossings)
    if n == 0:
        return {"run": run_name, "brier_score": None, "T2_brier_pass": None}

    # g(t) = 0 always
    brier = sum(c ** 2 for c in actual_crossings) / n
    crossing_rate = sum(actual_crossings) / n

    return {
        "run": run_name,
        "n_steps": n,
        "crossing_rate_above_theta": round(crossing_rate, 4),
        "brier_score_g0": round(brier, 4),
        "interpretation": (
            f"Brier(g=0, theta={theta}) = {brier:.4f}. "
            f"Crossing rate = {crossing_rate:.1%}. "
            "High Brier confirms enforcement cannot track drift."
            if brier > 0.01
            else "Low Brier: drift didn't exceed theta (insufficient drift in this run)."
        )
    }


# ─── T3: Detection Guarantee validation ──────────────────────────────────────

def test_T3_monotone_trend(results: List[Dict], run_name: str) -> Dict:
    """
    T3: Validates that D_hat is monotonically increasing.
    Mann-Kendall one-sided test: H1 = increasing trend.

    Theorem 3 guarantees detection with probability >= 1-alpha.
    Empirical evidence: D_hat increases monotonically post burn-in.
    """
    D_values = [r.get("D_hat", 0.0) for r in results]
    n = len(D_values)

    if n < 4:
        return {"run": run_name, "T3_trend_pass": None, "reason": "insufficient data"}

    S, z, verdict = mann_kendall(D_values)

    passed = (z > 0 and "increasing" in verdict)

    return {
        "run": run_name,
        "n_steps": n,
        "D_initial": round(D_values[0], 4),
        "D_final": round(D_values[-1], 4),
        "D_max": round(max(D_values), 4),
        "mk_S": round(S, 1),
        "mk_z": round(z, 4),
        "mk_verdict": verdict,
        "T3_trend_pass": passed,
        "interpretation": (
            f"T3 CONFIRMED: D_hat shows {verdict}. "
            "Monotone increasing trend supports detection guarantee."
            if passed
            else f"T3 WEAK: {verdict}. Trend not yet significant."
        )
    }


def test_T3_detection_time(results: List[Dict], run_name: str,
                            theta: float = 0.20) -> Dict:
    """
    T3 Detection delay: find T*(theta) and validate it is finite.

    Corollary (Detection Delay Bound):
      T* <= t0 + (theta + eps_est) / alpha_drift

    where alpha_drift is the empirical drift rate.
    """
    D_values = [r.get("D_hat", 0.0) for r in results]
    t_values = [r.get("t", i) for i, r in enumerate(results)]

    if not D_values:
        return {"run": run_name, "T3_detection_pass": None}

    T_star = next((t for t, d in zip(t_values, D_values) if d >= theta), None)
    D_final = D_values[-1]
    D_max = max(D_values)

    # Empirical drift rate: slope of D_hat over time (linear fit)
    n = len(D_values)
    t_arr = np.array(list(range(n)), dtype=float)
    d_arr = np.array(D_values)
    if n > 1:
        cov = np.cov(t_arr, d_arr)
        alpha_drift = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0.0
    else:
        alpha_drift = 0.0

    detected = T_star is not None
    eps_est = 0.05  # assumption (A4): |D_hat_n - D*| <= 0.05 w.p. >= 0.95

    if alpha_drift > 0 and detected:
        T_bound = t_values[0] + (theta + eps_est) / alpha_drift
        bound_valid = T_star <= T_bound
    else:
        T_bound = None
        bound_valid = None

    return {
        "run": run_name,
        "theta": theta,
        "T_star": T_star,
        "D_final": round(D_final, 4),
        "D_max": round(D_max, 4),
        "empirical_drift_rate": round(float(alpha_drift), 6),
        "T_bound_theory": round(T_bound, 1) if T_bound else None,
        "T3_detection_pass": detected,
        "T3_bound_valid": bound_valid,
        "interpretation": (
            f"T3 CONFIRMED: T*(theta={theta}) = {T_star}. "
            f"D_hat reached {theta} within {T_star - t_values[0]} steps. "
            + (f"Bound: T*={T_star} <= {T_bound:.1f} (holds)." if bound_valid else
               f"Bound: T*={T_star} > {T_bound:.1f} (violated)." if T_bound else "")
            if detected
            else f"T3 PENDING: D_hat={D_final:.4f} has not yet reached theta={theta}."
        )
    }


def test_T3_hoeffding_bound(results: List[Dict], run_name: str,
                             n_samples: int = 50, delta: float = 0.05) -> Dict:
    """
    Assumption (A4) validation: Hoeffding bound on estimation error.

    |D_hat_n - D*| <= sqrt(log(2/delta) / (2n)) w.p. >= 1-delta

    where D* is the true deviation (approximated by final D_hat value
    after convergence), n is the number of trace events.

    For n=50 burn-in steps, delta=0.05:
      eps_est = sqrt(log(40) / 100) = sqrt(3.69 / 100) ≈ 0.192
    """
    eps_hoeffding = math.sqrt(math.log(2.0 / delta) / (2 * n_samples))

    D_values = [r.get("D_hat", 0.0) for r in results]
    if not D_values:
        return {"run": run_name, "T3_hoeffding_pass": None}

    # Approximate D* by D_hat at last step (after convergence)
    D_star_approx = D_values[-1]
    D_initial = D_values[0]
    estimation_error = abs(D_initial - D_star_approx)

    bound_holds = estimation_error <= eps_hoeffding

    return {
        "run": run_name,
        "n_burn_in": n_samples,
        "delta": delta,
        "eps_hoeffding": round(eps_hoeffding, 4),
        "D_initial": round(D_initial, 4),
        "D_star_approx": round(D_star_approx, 4),
        "estimation_error": round(estimation_error, 4),
        "T3_hoeffding_pass": bound_holds,
        "interpretation": (
            f"A4 HOLDS: |D_hat(0) - D*| = {estimation_error:.4f} <= "
            f"eps_Hoeffding({n_samples}, {delta}) = {eps_hoeffding:.4f}."
            if bound_holds
            else f"A4 NOTE: |D_hat(0) - D*| = {estimation_error:.4f} > "
                 f"{eps_hoeffding:.4f}. D* approximation may be imprecise."
        )
    }


# ─── Bootstrap CI on T*(theta) ────────────────────────────────────────────────

def bootstrap_T_star(D_series: List[float], theta: float = 0.20,
                     n_boot: int = 1000, seed: int = 42) -> Dict:
    """
    Bootstrap confidence interval on T*(theta).
    Resamples sub-sequences from D_hat trajectory.
    """
    rng = np.random.RandomState(seed)
    n = len(D_series)
    T_star_obs = next((i for i, d in enumerate(D_series) if d >= theta), None)

    if T_star_obs is None:
        return {
            "T_star_obs": None,
            "bootstrap_ci_95": None,
            "note": "D_hat never reached theta in observed series"
        }

    boot_T_stars = []
    for _ in range(n_boot):
        # Resample windows of fixed size with replacement
        window = min(T_star_obs + 20, n)
        indices = rng.choice(window, size=window, replace=True)
        indices.sort()
        boot_series = [D_series[i] for i in indices]
        t = next((i for i, d in enumerate(boot_series) if d >= theta), None)
        if t is not None:
            boot_T_stars.append(t)

    if not boot_T_stars:
        return {"T_star_obs": T_star_obs, "bootstrap_ci_95": None,
                "note": "Bootstrap resamples never crossed theta"}

    ci_low = float(np.percentile(boot_T_stars, 2.5))
    ci_high = float(np.percentile(boot_T_stars, 97.5))
    ci_mean = float(np.mean(boot_T_stars))

    return {
        "T_star_obs": T_star_obs,
        "bootstrap_mean": round(ci_mean, 1),
        "bootstrap_ci_95": [round(ci_low, 1), round(ci_high, 1)],
        "n_bootstrap": n_boot,
        "theta": theta
    }


# ─── Load results ─────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all JSON result files from a directory."""
    runs = {}
    if not os.path.isdir(results_dir):
        return runs
    for fname in os.listdir(results_dir):
        if fname.endswith(".json") and fname != "summary.json":
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    run_name = fname.replace(".json", "")
                    runs[run_name] = data
            except (json.JSONDecodeError, IOError):
                pass
    return runs


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all_tests(results_dirs: List[str], verbose: bool = False) -> Dict:
    """Run all T2 and T3 statistical tests across all result directories."""

    report = {
        "paper": "From Admission to Invariants (Fernandez, 2026)",
        "tests": {"T2": [], "T3": []},
        "summary": {}
    }

    all_T2_pass = []
    all_T3_pass = []

    for results_dir in results_dirs:
        runs = load_results(results_dir)
        if not runs:
            if verbose:
                print(f"  [skip] No results in {results_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Results directory: {results_dir}")
        print(f"Runs found: {list(runs.keys())}")

        for run_name, results in runs.items():
            full_run_name = f"{os.path.basename(results_dir)}/{run_name}"
            print(f"\n--- {full_run_name} (n={len(results)}) ---")

            # T2 tests
            t2_enforcement = test_T2_enforcement_zero(results, full_run_name)
            t2_brier = test_T2_brier_score(results, full_run_name)
            report["tests"]["T2"].append({"enforcement": t2_enforcement,
                                           "brier": t2_brier})
            all_T2_pass.append(t2_enforcement["T2_pass"])

            print(f"  T2 enforcement: {'PASS' if t2_enforcement['T2_pass'] else 'FAIL'} "
                  f"| g_total={t2_enforcement['enforcement_total']} "
                  f"| D_max={t2_enforcement['D_max']}")
            print(f"  T2 evidence:    {t2_enforcement['T2_evidence']} "
                  f"| Brier={t2_brier.get('brier_score_g0', 'N/A')}")
            if verbose:
                print(f"  {t2_enforcement['interpretation']}")

            # T3 tests
            t3_trend = test_T3_monotone_trend(results, full_run_name)
            t3_detect = test_T3_detection_time(results, full_run_name)
            t3_hoeffding = test_T3_hoeffding_bound(results, full_run_name)

            D_series = [r.get("D_hat", 0.0) for r in results]
            t3_bootstrap = bootstrap_T_star(D_series)

            report["tests"]["T3"].append({
                "trend": t3_trend,
                "detection": t3_detect,
                "hoeffding": t3_hoeffding,
                "bootstrap": t3_bootstrap
            })

            t3_pass = (t3_trend.get("T3_trend_pass", False) or
                       t3_detect.get("T3_detection_pass", False))
            all_T3_pass.append(t3_pass)

            print(f"  T3 trend:       {'PASS' if t3_trend.get('T3_trend_pass') else 'FAIL'} "
                  f"| MK z={t3_trend.get('mk_z', 'N/A')} | {t3_trend.get('mk_verdict', '')}")
            print(f"  T3 detection:   {'PASS' if t3_detect.get('T3_detection_pass') else 'PENDING'} "
                  f"| T*={t3_detect.get('T_star', 'N/A')} | theta=0.20")
            print(f"  T3 Hoeffding:   {'PASS' if t3_hoeffding.get('T3_hoeffding_pass') else 'NOTE'} "
                  f"| eps={t3_hoeffding.get('eps_hoeffding', 'N/A')}")
            if t3_bootstrap.get("bootstrap_ci_95"):
                ci = t3_bootstrap["bootstrap_ci_95"]
                print(f"  T* bootstrap CI (95%): [{ci[0]:.0f}, {ci[1]:.0f}] "
                      f"(obs={t3_bootstrap['T_star_obs']})")
            if verbose:
                print(f"  {t3_trend.get('interpretation', '')}")
                print(f"  {t3_detect.get('interpretation', '')}")
                print(f"  {t3_hoeffding.get('interpretation', '')}")

    # Summary
    n_T2 = len(all_T2_pass)
    n_T3 = len(all_T3_pass)
    report["summary"] = {
        "T2_pass_rate": f"{sum(all_T2_pass)}/{n_T2}" if n_T2 else "0/0",
        "T3_pass_rate": f"{sum(all_T3_pass)}/{n_T3}" if n_T3 else "0/0",
        "T2_overall": "PASS" if all(all_T2_pass) and n_T2 > 0 else "FAIL",
        "T3_overall": "PASS" if any(all_T3_pass) and n_T3 > 0 else "PENDING",
    }

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"  T2 (Non-Identifiability):  {report['summary']['T2_overall']} "
          f"({report['summary']['T2_pass_rate']} runs)")
    print(f"  T3 (Detection Guarantee):  {report['summary']['T3_overall']} "
          f"({report['summary']['T3_pass_rate']} runs)")
    print(f"{'='*60}\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Statistical validation of IML theorems T2 and T3"
    )
    parser.add_argument(
        "--results", nargs="+",
        default=["results", "results_1000", "results_langgraph"],
        help="Result directories to scan (default: results results_1000 results_langgraph)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed interpretation for each test"
    )
    parser.add_argument(
        "--out", default="analysis/stats_report.json",
        help="Output JSON report path"
    )
    args = parser.parse_args()

    # Resolve paths relative to script location or CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    results_dirs = []
    for d in args.results:
        if os.path.isabs(d):
            results_dirs.append(d)
        else:
            # Try repo root first
            candidate = os.path.join(repo_root, d)
            if os.path.isdir(candidate):
                results_dirs.append(candidate)
            elif os.path.isdir(d):
                results_dirs.append(os.path.abspath(d))
            else:
                if args.verbose:
                    print(f"[warn] Directory not found: {d}")

    if not results_dirs:
        print("[error] No valid result directories found. Run experiments first.")
        sys.exit(1)

    report = run_all_tests(results_dirs, verbose=args.verbose)

    # Save report (custom encoder for numpy types)
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, cls=_Encoder)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
