# IML Benchmark

**Companion code for:**  
> Marcelo Fernandez (TraslaIA). *From Admission to Invariants: Measuring Deviation in Delegated Agent Systems.* 2026.  
> arXiv: [link TBD] · Paper 2 of the Agent Governance Trilogy

---

## What this is

This repository contains the full Python benchmark for the **Invariant Measurement Layer (IML)** — a monitoring layer that detects behavioral drift in autonomous agent systems *below* the enforcement boundary.

**The core result (Theorem 2):** No enforcement signal `g: Σ* → {0,1}` can recover whether an agent's behavior remains within its admission-time admissible space A₀. IML addresses this structural gap by anchoring deviation estimation to a frozen admission snapshot.

**Paper 1 (ACP):** https://github.com/chelof100/acp-framework-en  
**arXiv:** https://arxiv.org/abs/2603.18829

---

## Repository structure

```
iml-benchmark/
├── iml/                        # Core IML implementation
│   ├── deviation.py            # IML estimator (D̂ = 0.40·Dt + 0.35·Dc + 0.25·Dl)
│   ├── trace.py                # Trace data structure
│   └── snapshot.py             # AdmissionSnapshot (A₀ representation)
├── baselines/
│   ├── enforcement.py          # Enforcement signal g(τ) baseline
│   └── anomaly.py              # Rolling-window anomaly detector (B2)
├── runner/
│   ├── experiment.py           # Experiment runner
│   └── drift.py                # Drift injection (3 scenarios)
├── plots/
│   ├── plots.py                # Figures 1–4 (paper)
│   └── fig_longhorizon.py      # Figure 5: 1000-step validation
├── n8n_integration/
│   ├── iml_workflow_n8n.json   # Cloud-native n8n workflow (live webhook)
│   └── burn_in_generator.py    # Burn-in event generator
├── langgraph_experiment.py     # LangGraph agent experiment (§5.4)
└── main.py                     # Entry point
```

---

## Quick start

```bash
git clone https://github.com/chelof100/iml-benchmark
cd iml-benchmark
pip install -r requirements.txt
python main.py
```

**Reproduce all paper experiments:**
```bash
# Standard 300-step benchmark (T2 + T3 validation)
python main.py --steps 300 --seed 42

# Long-horizon 1000-step validation
python main.py --steps 1000 --seed 42 --output-dir results_1000

# Generate long-horizon figure (Fig. 5)
python plots/fig_longhorizon.py

# LangGraph agent experiment
python langgraph_experiment.py
```

---

## Key results (seed 42)

### 300-step benchmark

| Scenario | Enforcement | D̂ final | T*(θ=0.20) |
|---|---|---|---|
| Tool drift | **0** | 0.217 | t=256 |
| Delegation drift | **0** | 0.389 | t=130 |
| Context drift | **0** | 0.213 | t=258 |

### 1000-step long-horizon

| Scenario | Enforcement | D̂ final | T*(θ=0.20) |
|---|---|---|---|
| Tool drift | **0** | 0.229 | t=794 |
| Delegation drift | **0** | 0.393 | t=336 |
| Context drift | **0** | 0.227 | t=802 |
| **Total (3000 steps)** | **0** | — | — |

### Live n8n deployment (real agent traces, seed 99)

| Phase | Steps | Enforcement | D̂ final | T*(θ=0.30) |
|---|---|---|---|---|
| Baseline | 50 | **0** | 0.095 | — |
| Drift | 200 | **0** | 0.403 | t=9 |

---

## IML components

```
D̂(τ; A₀) = 0.40 · D_t(τ) + 0.35 · D_c(τ) + 0.25 · D_l(τ)
```

| Component | Formula | Measures |
|---|---|---|
| **D_t** | JS(P_τ ‖ P_{E₀}) | Tool distribution shift from admission |
| **D_c** | mean ρ(b) for b ∈ τ | Mean risk proximity to constraint boundary |
| **D_l** | norm. depth deviation | Delegation depth vs admission-time profile |

EMA smoothing: `D̂_t = 0.15 · D_raw + 0.85 · D̂_{t-1}`

---

## n8n live deployment

Webhook: `https://n8n.n8ncloud.top/webhook/iml-monitor`  
Workflow ID: `O1ZojC6kw6zW6RCf`

```bash
# Initialize A₀ (burn-in)
python n8n_integration/burn_in_generator.py

# Send a drift event
curl -X POST https://n8n.n8ncloud.top/webhook/iml-monitor \
  -H "Content-Type: application/json" \
  -d '{"action": "event", "agentId": "agent_001", "tool": "risky_delegate", "depth": 3}'
```

---

## Theoretical background

This benchmark empirically validates three formal results from the paper:

- **T1 (Existence):** `∃ τ ∈ g⁻¹(0)` with `τ ∉ A₀` — the compliance-invariance gap is non-empty
- **T2 (Non-Identifiability):** `A₀ ∉ σ(g)` — no function of the enforcement signal can recover A₀-membership
- **T3 (IML Recoverability):** IML is a consistent estimator of D(τ, A₀) with finite detection delay T*(θ)

---

## Part of the Agent Governance Series

| Paper | Title | Repo | Status |
|---|---|---|---|
| **Paper 0** | Atomic Decision Boundaries | [decision-boundary-model](https://github.com/chelof100/decision-boundary-model) | In preparation |
| **Paper 1** | Agent Control Protocol (ACP) | [acp-framework-en](https://github.com/chelof100/acp-framework-en) | [Published — arXiv:2603.18829](https://arxiv.org/abs/2603.18829) |
| **Paper 2** | From Admission to Invariants (this repo) | [iml-benchmark](https://github.com/chelof100/iml-benchmark) | In preparation |
| **Paper 3** | Fair Atomic Governance | [fair-atomic-governance](https://github.com/chelof100/fair-atomic-governance) | In preparation |
| **Paper 4** | Irreducible Multi-Scale Governance | [compositional-governance](https://github.com/chelof100/compositional-governance) | In preparation |

---

## Citation

```bibtex
@misc{fernandez2026iml,
  title   = {From Admission to Invariants: Measuring Deviation in Delegated Agent Systems},
  author  = {Fernandez, Marcelo},
  year    = {2026},
  note    = {arXiv: [TBD]. Companion code: https://github.com/chelof100/iml-benchmark}
}
```

---

## Author

**Marcelo Fernandez** · TraslaIA · info@traslaia.com  
https://agentcontrolprotocol.xyz
