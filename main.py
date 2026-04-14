"""
IML Benchmark — Entry Point
Runs all 3 drift scenarios and generates paper figures.

Usage:
    python main.py              # run experiments + generate figures
    python main.py --no-plots   # run experiments only
    python main.py --plots-only # generate figures from existing results
"""
import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="IML Benchmark")
    parser.add_argument("--steps", type=int, default=300,
                        help="Steps per scenario (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--plots-only", action="store_true",
                        help="Generate figures from existing results (skip experiments)")
    args = parser.parse_args()

    if not args.plots_only:
        from runner.experiment import run_all
        print("=" * 60)
        print("IML BENCHMARK — Empirical Validation of T2 and T3")
        print("=" * 60)
        print(f"  Steps: {args.steps}  |  Seed: {args.seed}  |  Output: {args.output_dir}/")
        print()

        summary = run_all(steps=args.steps, seed=args.seed, output_dir=args.output_dir)

        print()
        print("=" * 60)
        print("SUMMARY — Empirical Results")
        print("=" * 60)
        for mode, stats in summary.items():
            print(f"\n  [{mode}]")
            print(f"    D_hat final    = {stats['D_final']:.4f}")
            print(f"    D_hat max      = {stats['D_max']:.4f}")
            print(f"    D_hat @ t=150  = {stats['D_at_t150']}")
            print(f"    enforcement_triggers = {stats['enforcement_triggers']}  (T2: should be 0)")
            print(f"    T*(0.20)  = {stats['T_star_theta1']}  steps")
            print(f"    T*(0.40)  = {stats['T_star_theta2']}  steps")

        # T2 validation check
        print()
        print("-" * 60)
        all_g_zero = all(s["enforcement_triggers"] == 0 for s in summary.values())
        all_D_grows = all(
            s["D_final"] > 0.1 for s in summary.values()
        )
        print(f"  T2 VALIDATION (g=0 throughout): {'PASS' if all_g_zero else 'FAIL'}")
        print(f"  T3 VALIDATION (D_hat grows):    {'PASS' if all_D_grows else 'FAIL (check parameters)'}")
        print("-" * 60)

    if not args.no_plots:
        print()
        print("Generating figures...")
        try:
            from plots.plots import generate_all
            generate_all(results_dir=args.output_dir,
                         output_dir=os.path.join(args.output_dir, "figures"))
        except ImportError as e:
            print(f"  Warning: matplotlib not available — skipping plots ({e})")
        except FileNotFoundError as e:
            print(f"  Warning: results not found — run experiments first ({e})")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
