"""
Long-horizon figure (1000 steps) for paper §5.4.
Shows D̂ growing monotonically while g(τ) = 0 throughout.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "tool_drift":       "#E63946",
    "delegation_drift": "#457B9D",
    "context_drift":    "#2A9D8F",
}
LABELS = {
    "tool_drift":       "Tool drift",
    "delegation_drift": "Delegation drift",
    "context_drift":    "Context drift",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_1000")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4))

for mode in ["tool_drift", "delegation_drift", "context_drift"]:
    with open(os.path.join(RESULTS_DIR, f"{mode}.json")) as f:
        data = json.load(f)
    ts     = [r["t"] for r in data]
    D_hats = [r["D_hat"] for r in data]
    ax.plot(ts, D_hats, label=LABELS[mode], color=COLORS[mode], linewidth=2)

# Enforcement flat line
ax.axhline(y=0, color="#F4A261", linestyle=":", linewidth=1.8,
           label=r"Enforcement $g(\tau_t) = 0$ (all scenarios)", alpha=0.9)

# Drift onset
ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.annotate("drift onset", xy=(52, 0.01), fontsize=8, color="gray")

# Detection threshold reference
ax.axhline(y=0.20, color="#6c757d", linestyle=(0, (3,5)), linewidth=1.2,
           label=r"Detection threshold $\theta = 0.20$", alpha=0.7)

ax.set_xlabel("Step $t$")
ax.set_ylabel(r"$\widehat{D}(\tau_t,\,\mathcal{A}_0)$")
ax.set_title(r"Long-horizon validation (1000 steps): $\widehat{D}$ grows, $g(\tau) = 0$ throughout")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_xlim(0, 1000)
ax.set_ylim(-0.02, 0.55)

fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"fig5_longhorizon.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"Saved: {out}")

# Print stats for paper text
for mode in ["tool_drift", "delegation_drift", "context_drift"]:
    with open(os.path.join(RESULTS_DIR, f"{mode}.json")) as f:
        data = json.load(f)
    D_final = data[-1]["D_hat"]
    enf     = sum(r["enforcement"] for r in data)
    T_star  = next((r["t"] for r in data if r["D_hat"] >= 0.20), None)
    print(f"  {mode}: D_final={D_final:.4f}  enforcement={enf}  T*(0.20)={T_star}")
