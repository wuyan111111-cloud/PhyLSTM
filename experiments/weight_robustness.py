"""
Experiment 5 — Robustness Under Different Decision Weights (Table S6).

Evaluates PhyLSTM's ranking stability under six weight configurations
corresponding to different stakeholder preference profiles.

RSSI is computed via propagate_prediction_error: each model's prediction
error is first propagated into strategy-concentration space, so that models
with lower errors (PhyLSTM) start from a more faithful baseline before
additional 5% noise perturbations are applied.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from data.generate_data import (build_dataset, get_train_val_test_split,
                                 normalize, N_SECTIONS, N_TIMESTEPS)
from models import PlainLSTM, PhyLSTM
from utils import (Trainer, make_dataset, DEVICE,
                   propagate_prediction_error,
                   compute_rssi, aggregate_rssi,
                   generate_perturbations, compute_decision_scores,
                   BASE_STRATEGIES, WEIGHT_CONFIGS)

INPUT_DIM    = 6
N_RUNS       = 10
N_PERT       = 50
K_BASELINE   = 0.05
V_IDX, D_IDX = 0, 5

STRATEGY_KS = [s.k for s in BASE_STRATEGIES]   # [0.05, 0.12, 0.25]


def single_weight_run(dataset: dict, seed: int) -> dict:
    rng        = np.random.default_rng(seed)
    features   = dataset["features"]
    targets    = dataset["targets"]
    strategy_C = dataset["strategy_C"]

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = get_train_val_test_split(
        features, targets)
    T_val_end = int(N_TIMESTEPS * 0.8)
    sC_te = strategy_C[:, T_val_end:, :]    # (8, T_test, 3)

    X_tr_n, mu, sigma = normalize(X_tr)
    X_v_n,  _, _      = normalize(X_v,  mu, sigma)
    X_te_n, _, _      = normalize(X_te, mu, sigma)

    v_tr = X_tr_n[:, :, V_IDX]; D_tr = X_tr_n[:, :, D_IDX]
    v_v  = X_v_n[:, :, V_IDX];  D_v  = X_v_n[:, :, D_IDX]

    train_ds = make_dataset(X_tr_n, y_tr, v_tr, D_tr)
    val_ds   = make_dataset(X_v_n,  y_v,  v_v,  D_v)
    costs    = [s.cost_dict() for s in BASE_STRATEGIES]
    C0       = float(targets.max())

    results = {}

    for model_name, model in [
        ("Plain LSTM", PlainLSTM(INPUT_DIM, 64, 2, dropout=0.2)),
        ("PhyLSTM",    PhyLSTM(INPUT_DIM, 64, 2, dropout=0.2,
                                lambda_phy=0.1, lambda_bc=0.05)),
    ]:
        trainer = Trainer(model, lr=0.001, max_epochs=500,
                          patience=50, k_decay=K_BASELINE)
        trainer.fit(train_ds, val_ds)
        y_pred = trainer.predict(X_te_n)    # (8, T_test)

        weight_rssi: dict[str, list[float]] = {wn: [] for wn in WEIGHT_CONFIGS}

        for sec in range(N_SECTIONS):
            # Propagate model prediction error into strategy concentrations
            pred_error  = y_pred[sec] - y_te[sec]
            pred_sC_nom = propagate_prediction_error(
                sC_te[sec], pred_error, STRATEGY_KS, K_BASELINE)  # (T_test, 3)
            pred_sC_pert = generate_perturbations(
                pred_sC_nom, n_pert=N_PERT, noise_pct=0.15, rng=rng)

            for wname, w in WEIGHT_CONFIGS.items():
                D_nom  = compute_decision_scores(pred_sC_nom,  costs,
                                                  weights=w, C0=C0)
                D_pert = np.stack([
                    compute_decision_scores(pred_sC_pert[k], costs,
                                            weights=w, C0=C0)
                    for k in range(N_PERT)
                ])
                weight_rssi[wname].append(
                    aggregate_rssi(compute_rssi(D_nom, D_pert)))

        results[model_name] = {wn: float(np.mean(v))
                               for wn, v in weight_rssi.items()}

    return results


def run_weight_robustness(n_runs: int = N_RUNS):
    print("\n" + "=" * 60)
    print("Experiment: Weight Robustness Analysis (Table S6)")
    print(f"  Runs: {n_runs}")
    print("=" * 60)

    dataset = build_dataset(seed=42)
    agg = {
        "Plain LSTM": {wn: [] for wn in WEIGHT_CONFIGS},
        "PhyLSTM":    {wn: [] for wn in WEIGHT_CONFIGS},
    }

    for run in range(n_runs):
        if run % 3 == 0:
            print(f"  run {run + 1}/{n_runs} …")
        res = single_weight_run(dataset, seed=run)
        for m_name in ["Plain LSTM", "PhyLSTM"]:
            for wn in WEIGHT_CONFIGS:
                agg[m_name][wn].append(res[m_name][wn])

    # ── Table S6 ──────────────────────────────────────────────────────────
    header = (f"{'Weight Config':<25} "
              f"{'w_eff':>6} {'w_cost':>7} {'w_eco':>6} {'w_risk':>7} "
              f"{'Plain LSTM RII':>16} {'PhyLSTM RII':>13} {'RII Reduction':>15}")
    print("\n" + "=" * len(header))
    print("Table S6 — Ranking Stability Under Different Decision Weights")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for wname, w in WEIGHT_CONFIGS.items():
        plain_rssi = float(np.mean(agg["Plain LSTM"][wname]))
        phy_rssi   = float(np.mean(agg["PhyLSTM"][wname]))
        plain_rii  = 1.0 - plain_rssi
        phy_rii    = 1.0 - phy_rssi
        reduction  = (plain_rii - phy_rii) / (plain_rii + 1e-9) * 100

        print(f"{wname:<25} "
              f" {w[0]:.2f}  {w[1]:.2f}  {w[2]:.2f}  {w[3]:.2f}   "
              f"  {plain_rii:.3f}          "
              f"  {phy_rii:.3f}      "
              f"  {reduction:.0f}%")

    print("=" * len(header))
    print("Note: RII = 1 − RSSI (lower = more stable). "
          "Each model's prediction error is propagated into "
          "strategy concentrations before scoring.\n")


if __name__ == "__main__":
    run_weight_robustness(n_runs=5)
