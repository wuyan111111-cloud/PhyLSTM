# PhyLSTM: Physics-Guided Sequential Learning for Ecological Restoration Decision Stability

> **Paper**: *Improving Decision Stability in Ecological Restoration Planning Using Physics-Guided Sequential Learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository provides a full reproduction of the **PhyLSTM** framework proposed in the paper. PhyLSTM embeds mass conservation and reaction kinetics as **soft regularization terms** (via a lumped temporal mass balance residual derived from the Advection-Diffusion-Reaction equation) into a recurrent neural network, stabilizing decision-relevant trajectory dynamics without sacrificing temporal memory.

### Key Contributions

| Contribution | Description |
|---|---|
| **PhyLSTM** | Physics-guided LSTM with lumped mass-balance residual soft penalty |
| **RSSI** | Ranking Stability Sensitivity Index (Kendall's τ-based, N_pert = 1 000) |
| **Accuracy–Stability Decoupling** | RMSE ↔ RSSI are structurally decoupled; 19% RSSI gain with <7% RMSE difference at 10% data |
| **MBE Anchoring** | Mass Balance Error < 0.01 throughout vs 0.048 for plain LSTM |
| **Inductive-Bias Superiority** | Physics regularization achieves 3.2× greater RII reduction than matched-strength L2 regularization |

---

## Repository Structure

```
PhyLSTM/
├── main.py                          # Entry point: run all experiments
├── requirements.txt
│
├── data/
│   └── generate_data.py             # Synthetic ADR-based dataset (960 samples)
│
├── models/
│   ├── lstm.py                      # Plain LSTM baseline
│   ├── feature_lstm.py              # Feature-augmented LSTM
│   ├── pinn.py                      # Standard PINN (feedforward)
│   └── phylstm.py                   # PhyLSTM (proposed)
│
├── utils/
│   ├── metrics.py                   # RMSE, RSSI (N_pert=1000), MBE, RII
│   ├── strategies.py                # Restoration strategy definitions (A/B/C)
│   ├── trainer.py                   # Unified training loop
│   └── visualization.py            # Figures 2–6
│
├── experiments/
│   ├── data_scarcity.py             # Table 3 / Table S4, Figures 2–5
│   ├── ablation.py                  # Table 4a (main text) — ablation study
│   ├── sensitivity.py               # Table 4  — parameter sensitivity
│   ├── six_strategy.py              # Table 5  — six-strategy expansion
│   └── weight_robustness.py         # Table S7 — weight robustness
│
└── figures/                         # Auto-generated output figures
```

---

## Installation

```bash
git clone https://github.com/<your-username>/PhyLSTM.git
cd PhyLSTM
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4

**Hardware used in paper:** Intel Xeon Gold 6248R, NVIDIA RTX 3090 (24 GB), 128 GB RAM.  
The code runs on CPU as well (slower; expect ~10× longer runtime).

---

## Quick Start

```bash
# Smoke-test with 3 runs (fast, ~5 min on CPU)
python main.py --quick

# Full replication: all experiments, 20 independent runs (~45 min on RTX 3090)
python main.py

# Individual experiments
python main.py --experiment data        # Table 3/S4, Figures 2–5
python main.py --experiment ablation    # Table 4a — ablation study
python main.py --experiment sensitivity # Table 4   — parameter sensitivity
python main.py --experiment six_strat   # Table 5   — 6-strategy expansion
python main.py --experiment weight      # Table S7  — weight robustness
```

---

## Framework

### PhyLSTM Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phy}} \mathcal{L}_{\text{physics}} + \lambda_{\text{bc}} \mathcal{L}_{\text{boundary}}$$

where $\mathcal{L}_{\text{physics}} = \frac{1}{T}\sum_{t=1}^{T} r_t^2$.

### Lumped Temporal Mass Balance Residual (Eq. S3)

$$r_t = \frac{\hat{C}_t - \hat{C}_{t-1}}{\Delta t} + k\hat{C}_t - \hat{F}_t$$

$$\hat{F}_t = w_v \tilde{v}_t + w_Q \tilde{Q}_t + w_D \tilde{D}_t$$

where $\tilde{v}_t, \tilde{Q}_t, \tilde{D}_t$ are standardized inputs (zero-mean, unit-variance); $\{w_v, w_Q, w_D\}$ are **learnable scalar parameters** initialized from physically derived benchmarks:

$$w_{v,\text{theory}} = -\frac{\bar{C}\,\sigma_v}{L}, \quad
  w_{Q,\text{theory}} = \frac{(\bar{C}_{\text{in}}-\bar{C})\,\sigma_Q}{L\cdot A}, \quad
  w_{D,\text{theory}} = \frac{\bar{C}\,\sigma_D}{L^2}$$

All three carry units of mg L⁻¹ s⁻¹ (see SI S1.3 for dimensional verification). **No spatial grid step Δx is required**; spatial derivatives are integrated over the reach length $L$. The formulation is applied independently to each of the eight monitoring stations.

> **Physical validity check (SI Table S1):** After training, learned weights remain within a mean absolute deviation of 10.3 % ± 2.8 % from their theoretical benchmarks across all eight stations and three weight dimensions, confirming physical interpretability.

### RSSI Definition (Section 2.2)

$$\text{RSSI}_t = \frac{1}{N_{\text{pert}}} \sum_{k=1}^{N_{\text{pert}}} \tau\!\left(\pi_t^{(0)},\, \pi_t^{(\delta^{(k)})}\right)$$

where $\tau(\cdot)$ is Kendall's rank correlation coefficient, $N_{\text{pert}} = 1{,}000$ perturbation scenarios are drawn from a $\pm10\%$ uniform distribution applied simultaneously to all input dimensions, and 95% confidence intervals are computed via bootstrap resampling over the $N_{\text{pert}}$ scenarios.

---

## Hyperparameters (SI Table S2)

| Parameter | Plain LSTM | Feature-Aug LSTM | Standard PINN | PhyLSTM |
|---|---|---|---|---|
| Hidden dim | 64 | 64 | [128, 64, 32] | 64 |
| LSTM layers | 2 | 2 | — | 2 |
| Dropout | 0.2 | 0.2 | 0.1 | 0.2 |
| Learning rate | 0.001 | 0.001 | 0.0005 | 0.001 |
| Max epochs | 500 | 500 | 1 000 | 500 |
| Early stopping patience | 50 | 50 | 100 | 50 |
| λ_phy | — | — | 1.0 | **0.1** |
| λ_bc | — | — | 0.5 | **0.05** |

Validation metric for hyperparameter search: $0.3 \times \text{RMSE} + 0.7 \times (1 - \text{Kendall's}\,\tau)$.

---

## Restoration Strategies (Table 2)

| Strategy | Name | k (d⁻¹) | D (m²/s) | Init Cost (¥10⁴) | O&M (¥10⁴/yr) |
|---|---|---|---|---|---|
| A | Natural Attenuation | 0.05 | 0.008 | 0 | 0 |
| B | Phytoremediation | 0.12 | 0.010 | 120 | 15 |
| C | Active Replacement | 0.25 | 0.012 | 580 | 45 |

Strategy C has discrete state resets at *t* = 30, 60, 90 months (α = 0.20, 0.40, 0.50 respectively). Note: *k* is given in d⁻¹; the code converts to month⁻¹ (multiply by 30.44) for ADR residual calculation.

---

## Key Results

### Accuracy–Stability Decoupling (Table 3)

| Model | 10% RMSE | 10% RSSI | 100% RSSI |
|---|---|---|---|
| Plain LSTM | 0.134 ± 0.008 | 0.746 ± 0.031 | 0.844 ± 0.018 |
| Feature-Aug LSTM | 0.128 ± 0.007 | 0.782 ± 0.025 | 0.868 ± 0.015 |
| Standard PINN | 0.145 ± 0.011 | 0.811 ± 0.022 | 0.859 ± 0.016 |
| **PhyLSTM (ours)** | **0.125 ± 0.006** | **0.888 ± 0.012** | **0.922 ± 0.009** |

PhyLSTM improves RSSI by ~19% over plain LSTM at 10% data, with <7% difference in RMSE.

### Ablation Study (Table 4a, main text)

| Configuration | RMSE | RII | MBE (t=100) |
|---|---|---|---|
| PhyLSTM Full | 0.082 ± 0.003 | 0.078 ± 0.009 | 0.009 ± 0.003 |
| Disable ℒ_phy | 0.080 ± 0.003 | 0.131 ± 0.016 | 0.042 ± 0.009 |
| Disable ℒ_bc | 0.083 ± 0.004 | 0.095 ± 0.011 | 0.012 ± 0.004 |
| Disable All (Pure LSTM) | 0.089 ± 0.004 | 0.156 ± 0.018 | 0.048 ± 0.010 |
| L2-LSTM† | 0.083 ± 0.004 | 0.132 ± 0.015 | 0.044 ± 0.009 |

RII = 1 − τ, lower is better.  
†L2-LSTM: Plain LSTM + L2 regularization at matched penalty magnitude (λ_L2 = 8×10⁻⁴). PhyLSTM achieves 3.2× greater RII reduction than L2-LSTM at identical regularization strength, confirming that physics provides a strictly more informative inductive bias than norm-based shrinkage.

---

## Reproducibility

**Random seeds** (20 independent runs):
```
[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021,
 2223, 2425, 2627, 2829, 3031, 3233, 3435, 3637, 3839, 4041]
```

**Data:** Raw monitoring data are confidential (Henan Provincial Dept. of Ecology and Environment). A **synthetic benchmark dataset** with the same statistical properties is provided at `data/synthetic_henan.csv` to verify code correctness and reproduce relative performance comparisons (Figures 3 and 5).

**Docker:**
```bash
docker build -t phylstm .
docker run --rm phylstm python main.py --quick
```

**Expected runtime:** ~45 min for full 20-run replication on NVIDIA RTX 3090; ~8–10 h on CPU.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{wu2025phylstm,
  title   = {Improving Decision Stability in Ecological Restoration Planning
             Using Physics-Guided Sequential Learning},
  author  = {Wu, Yan and Wang, Caifeng and Wu, Hang and Wu, Wanying and
             Tian, Weili and Zhao, Can and Wang, Kai},
  journal = {PNAS Nexus},
  year    = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
