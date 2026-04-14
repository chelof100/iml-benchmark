"""
Plots — Figure generation for the IML paper.
Produces 4 figures:
  Fig 1: D̂(τ_t) drift curves for all 3 scenarios
  Fig 2: Component breakdown (D_t, D_c, D_l) vs enforcement g(τ)
  Fig 3: IML vs anomaly detection score comparison
  Fig 4: T*(θ) detection delay across thresholds
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List


# ─── Style ────────────────────────────────────────────────────────────────────
COLORS = {
    "tool_drift":        "#E63946",   # red
    "delegation_drift":  "#457B9D",   # blue
    "context_drift":     "#2A9D8F",   # teal
    "enforcement":       "#F4A261",   # orange
    "anomaly":           "#A8DADC",   # light blue
    "iml":               "#264653",   # dark teal
}

MODE_LABELS = {
    "tool_drift":       "Tool drift",
    "delegation_drift": "Delegation drift",
    "context_drift":    "Context drift",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def load_results(results_dir: str) -> Dict[str, List[Dict]]:
    data = {}
    for mode in ["tool_drift", "delegation_drift", "context_drift"]:
        path = os.path.join(results_dir, f"{mode}.json")
        with open(path) as f:
            data[mode] = json.load(f)
    return data


# ─── Figure 1: D̂(τ_t) for all 3 scenarios ──────────────────────────────────

def fig_drift_curves(data: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))

    for mode, results in data.items():
        ts = [r["t"] for r in results]
        D_hats = [r["D_hat"] for r in results]
        enforcements = [r["enforcement"] for r in results]

        ax.plot(ts, D_hats, label=MODE_LABELS[mode], color=COLORS[mode], linewidth=2)

        # Mark drift start (t=50)
        ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Enforcement is always 0 — show as flat line
    ax.axhline(y=0, color=COLORS["enforcement"], linestyle=":", linewidth=1.5,
               label="g(τ) = 0 (all scenarios)", alpha=0.8)

    ax.set_xlabel("Step t")
    ax.set_ylabel(r"$\hat{D}(\tau_t, A_0)$")
    ax.set_title(r"IML Drift Signal: $\hat{D}$ grows while $g(\tau) = 0$")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.02, 1.0)
    ax.annotate("drift start", xy=(50, 0.02), fontsize=8, color="gray")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_drift_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    path_png = os.path.join(output_dir, "fig1_drift_curves.png")
    fig.savefig(path_png, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# ─── Figure 2: Component breakdown vs enforcement ──────────────────────────

def fig_component_breakdown(data: Dict, output_dir: str):
    modes = list(data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, mode in zip(axes, modes):
        results = data[mode]
        ts = [r["t"] for r in results]
        D_t = [r["D_t"] for r in results]
        D_c = [r["D_c"] for r in results]
        D_l = [r["D_l"] for r in results]
        D_hat = [r["D_hat"] for r in results]
        g = [r["enforcement"] for r in results]

        ax.plot(ts, D_hat, label=r"$\hat{D}$", color=COLORS[mode], linewidth=2)
        ax.plot(ts, D_t, label=r"$D_t$", color="#6D6875", linewidth=1, linestyle="--")
        ax.plot(ts, D_c, label=r"$D_c$", color="#B5838D", linewidth=1, linestyle="-.")
        ax.plot(ts, D_l, label=r"$D_l$", color="#FFCDB2", linewidth=1, linestyle=":")
        ax.axhline(y=0, color=COLORS["enforcement"], linestyle=":", linewidth=1.5,
                   alpha=0.7, label=r"$g(\tau)=0$")
        ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Step t")
        if ax == axes[0]:
            ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.1)

    fig.suptitle("Component Breakdown: IML vs Enforcement", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_component_breakdown.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig2_component_breakdown.png"), bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# ─── Figure 3: IML vs Anomaly Detector ─────────────────────────────────────

def fig_iml_vs_anomaly(data: Dict, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, mode in zip(axes, list(data.keys())):
        results = data[mode]
        ts = [r["t"] for r in results]
        D_hat = [r["D_hat"] for r in results]
        anomaly = [r["anomaly"] for r in results]

        ax.plot(ts, D_hat, label=r"IML $\hat{D}$", color=COLORS[mode], linewidth=2)
        ax.plot(ts, anomaly, label="Anomaly (B2)", color=COLORS["anomaly"],
                linewidth=1.5, linestyle="--")
        ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Step t")
        if ax == axes[0]:
            ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.02, 1.0)

    fig.suptitle("IML vs Anomaly Detection: Normative vs Statistical Reference", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_iml_vs_anomaly.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig3_iml_vs_anomaly.png"), bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# ─── Figure 4: Detection delay T*(θ) ────────────────────────────────────────

def fig_detection_delay(data: Dict, output_dir: str):
    thresholds = np.linspace(0.05, 0.80, 50)
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode, results in data.items():
        D_hats = [r["D_hat"] for r in results]
        delays = []
        for theta in thresholds:
            t_star = next((i for i, d in enumerate(D_hats) if d > theta), None)
            delays.append(t_star if t_star is not None else len(D_hats))

        ax.plot(thresholds, delays, label=MODE_LABELS[mode],
                color=COLORS[mode], linewidth=2)

    ax.set_xlabel(r"Detection threshold $\theta$")
    ax.set_ylabel(r"$T^*(\theta)$ — detection delay (steps)")
    ax.set_title(r"Detection Delay $T^*(\theta)$ as a Function of Threshold")
    ax.legend()
    ax.set_xlim(0.05, 0.80)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_detection_delay.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig4_detection_delay.png"), bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# ─── Entry point ─────────────────────────────────────────────────────────────

def generate_all(results_dir: str = "results", output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading results...")
    data = load_results(results_dir)

    print("Generating figures...")
    fig_drift_curves(data, output_dir)
    fig_component_breakdown(data, output_dir)
    fig_iml_vs_anomaly(data, output_dir)
    fig_detection_delay(data, output_dir)

    print(f"\nAll figures saved to {output_dir}/")
