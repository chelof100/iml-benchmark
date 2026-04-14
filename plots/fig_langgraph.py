"""Figure: LangGraph agent drift experiment."""
import json, os
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.labelsize": 12, "figure.dpi": 150})

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results_langgraph", "langgraph_drift.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)

ts    = [r["t"] for r in data]
D_hat = [r["D_hat"] for r in data]
D_t   = [r["D_t"]   for r in data]
D_c   = [r["D_c"]   for r in data]
D_l   = [r["D_l"]   for r in data]
enf   = [r["enforcement"] for r in data]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts, D_hat, color="#264653", linewidth=2.5, label=r"IML $\widehat{D}$ (composite)")
ax.plot(ts, D_t,   color="#E63946", linewidth=1.2, linestyle="--",  label=r"$D_t$ (tool distribution)")
ax.plot(ts, D_c,   color="#457B9D", linewidth=1.2, linestyle="-.",  label=r"$D_c$ (constraint proximity)")
ax.plot(ts, D_l,   color="#2A9D8F", linewidth=1.2, linestyle=":",   label=r"$D_l$ (lineage depth)")
ax.axhline(y=0,    color="#F4A261", linewidth=1.8, linestyle=":",
           label=r"Enforcement $g(\tau_t) = 0$", alpha=0.9)
ax.axhline(y=0.20, color="#6c757d", linewidth=1.0, linestyle=(0,(3,5)),
           label=r"$\theta = 0.20$", alpha=0.7)
ax.axvline(x=50,   color="gray",    linewidth=0.8, linestyle="--", alpha=0.5)
ax.annotate("drift onset", xy=(52, 0.005), fontsize=8, color="gray")

ax.set_xlabel("Step $t$")
ax.set_ylabel(r"Deviation score")
ax.set_title("LangGraph agent: compliant drift detected by IML")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.set_xlim(50, max(ts))
ax.set_ylim(-0.02, 0.55)
fig.tight_layout()

for ext in ("pdf", "png"):
    p = os.path.join(OUT_DIR, f"fig6_langgraph.{ext}")
    fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"Saved: {p}")

D_final = data[-1]["D_hat"]
enf_sum = sum(r["enforcement"] for r in data)
T_star  = next((r["t"] for r in data if r["D_hat"] >= 0.20), None)
print(f"D_final={D_final:.4f}  enforcement={enf_sum}  T*(0.20)={T_star}")
