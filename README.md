# PhyLSTM: Physics-Guided Sequential Learning for Ecological Restoration Decision Stability

> **Paper**: *From Prediction Accuracy to Decision Stability: Enhancing Robustness in Ecological Restoration Planning via Physics-Guided Sequential Learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository provides a full reproduction of the **PhyLSTM** framework proposed in the paper. PhyLSTM embeds mass conservation and reaction kinetics (via the Advection-Diffusion-Reaction equation) as **soft regularization terms** into a recurrent neural network, stabilizing decision-relevant trajectory dynamics without sacrificing temporal memory.

### Key Contributions

| Contribution | Description |
|---|---|
| **PhyLSTM** | Physics-guided LSTM with ADR residual soft penalty |
| **RSSI** | Ranking Stability Sensitivity Index (Kendall's τ-based) |
| **Accuracy–Stability Decoupling** | RMSE ↔ RSSI are structurally decoupled; 19% RSSI gain with <7% RMSE difference at 10% data |
| **MBE Anchoring** | Mass Balance Error < 0.01 mg L⁻¹ throughout vs 0.048 mg L⁻¹ for plain LSTM |

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
│   ├── metrics.py                   # RMSE, RSSI, MBE, RII
│   ├── strategies.py                # Restoration strategy definitions
│   ├── trainer.py                   # Unified training loop
│   └── visualization.py            # Figures 1–4
│
├── experiments/
│   ├── data_scarcity.py             # Table 3/S3, Figures 1–3
│   ├── ablation.py                  # Table S5
│   ├── sensitivity.py               # Table 4
│   ├── six_strategy.py              # Table 5, Figure 4
│   └── weight_robustness.py         # Table S6
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
The code runs on CPU as well (slower).

---

## Quick Start

```bash
# Smoke-test with 3 runs (fast, ~5 min on CPU)
python main.py --quick

# Full replication: all experiments, 20 independent runs
python main.py

# Individual experiments
python main.py --experiment data        # Table 3/S3, Figures 1–3
python main.py --experiment ablation    # Table S5 – ablation study
python main.py --experiment sensitivity # Table 4  – parameter sensitivity
python main.py --experiment six_strat   # Table 5, Figure 4 – 6-strategy expansion
python main.py --experiment weight      # Table S6 – weight robustness
```

---

## Framework

### PhyLSTM Loss Function (Eq. S4)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phy}} \mathcal{L}_{\text{physics}} + \lambda_{\text{bc}} \mathcal{L}_{\text{boundary}}$$

### ADR Physics Residual (Eq. S2)

$$r_t = \hat{C}_t - \hat{C}_{t-1} + \Delta t \left[ v_t \frac{\hat{C}_t - \hat{C}_{t-1}}{\Delta x} - D_t \frac{\hat{C}_{t+1} - 2\hat{C}_t + \hat{C}_{t-1}}{\Delta x^2} + k\hat{C}_t \right]$$

### RSSI Definition (Section 2.2)

$$\text{RSSI}_t = \frac{1}{N_{\text{pert}}} \sum_{k=1}^{N_{\text{pert}}} \tau\!\left(\pi_t^{(0)},\, \pi_t^{(\delta^{(k)})}\right)$$

where $\tau(\cdot)$ is Kendall's rank correlation coefficient.

---

## Hyperparameters (Table S1)

| Parameter | Plain LSTM | Feature-Aug LSTM | Standard PINN | PhyLSTM |
|---|---|---|---|---|
| Hidden dim | 64 | 64 | [128,64,32] | 64 |
| LSTM layers | 2 | 2 | — | 2 |
| Dropout | 0.2 | 0.2 | 0.1 | 0.2 |
| Learning rate | 0.001 | 0.001 | 0.0005 | 0.001 |
| Max epochs | 500 | 500 | 1000 | 500 |
| Early stopping | 50 | 50 | 100 | 50 |
| λ_phy | — | — | 1.0 | **0.1** |
| λ_bc | — | — | 0.5 | **0.05** |

---

## Restoration Strategies (Table 2)

| Strategy | Name | k (d⁻¹) | D (m²/s) | Init Cost (¥10⁴) | O&M (¥10⁴/yr) |
|---|---|---|---|---|---|
| A | Natural Attenuation | 0.05 | 0.008 | 0 | 0 |
| B | Phytoremediation | 0.12 | 0.010 | 120 | 15 |
| C | Active Replacement | 0.25 | 0.012 | 580 | 45 |

Strategy C has discrete state resets at *t* = 30, 60, 90 months (α = 0.20, 0.40, 0.50).

---

## Key Results (Table 3)

| Model | 10% RMSE | 10% RSSI | 100% RSSI |
|---|---|---|---|
| Plain LSTM | 0.134±0.008 | 0.746±0.031 | 0.844±0.018 |
| Feature-Aug LSTM | 0.128±0.007 | 0.782±0.025 | 0.868±0.015 |
| Standard PINN | 0.145±0.011 | 0.811±0.022 | 0.859±0.016 |
| **PhyLSTM (ours)** | **0.125±0.006** | **0.888±0.012** | **0.922±0.009** |

**PhyLSTM improves RSSI by ~19% over plain LSTM at 10% data, with <7% difference in RMSE.**

---

## Generated Figures

| Figure | File | Generated by |
|---|---|---|
| Fig. 1 | `figures/fig1_rmse_rssi_vs_data.png` | `experiments/data_scarcity.py` |
| Fig. 2 | `figures/fig2_kendall_tau_and_mbe.png` | `experiments/data_scarcity.py` |
| Fig. 3 | `figures/fig3_rii_boxplots.png` | `experiments/data_scarcity.py` |
| Fig. 4 | `figures/fig4_six_strategy_kendall_tau.png` | `experiments/six_strategy.py` |

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{wu2025phylstm,
  title   = {From Prediction Accuracy to Decision Stability: Enhancing Robustness
             in Ecological Restoration Planning via Physics-Guided Sequential Learning},
  author  = {Wu, Yan and Wang, Caifeng and Wu, Hang and Wu, Wanying and
             Tian, Weili and Zhao, Can and Wang, Kai},
  journal = {PNAS Nexus},
  year    = {2025},
  note    = {under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
