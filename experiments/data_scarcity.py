"""
Experiment 1 — Data Scarcity Analysis (Section 4, main text).

Reproduces:
  - Table 3 / Table S3 : RMSE and RSSI under 5/10/20/50/100% training data
  - Figure 1 : RMSE and RSSI vs training-data proportion (two-panel)
  - Figure 2 : (a) Temporal Kendall's τ; (b) Cumulative MBE over 120 months
  - Figure 3 : Box plots of Ranking Instability Index (RII = 1 − τ)

RSSI computation (correct model-dependent logic):
  1. Model predicts y_pred for test inputs.
  2. Compute prediction error ε = y_pred − y_true per section.
  3. Propagate ε into per-strategy concentrations via k-ratio scaling
     (propagate_prediction_error). Different k ratios cause strategies to
     respond differently to the same prediction error, creating real ranking
     changes when accuracy is low.
  4. Add 5% Gaussian perturbations to obtain D_pert.
  5. RSSI_t = Kendall τ(rank(D_nom_t), rank(D_pert_t)).
  Physics interpretation: PhyLSTM's mass-balance regularisation suppresses
  error accumulation, so its pred_sC stays close to true sC → stable rankings.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from data.generate_data import (build_dataset, get_train_val_test_split,
                                 subsample_training_data, normalize,
                                 N_SECTIONS, N_TIMESTEPS, STRATEGY_PARAMS)
from models import PlainLSTM, FeatureAugLSTM, StandardPINN, PhyLSTM
from utils  import (Trainer, make_dataset, DEVICE,
                    propagate_prediction_error,
                    compute_rssi, aggregate_rssi, compute_mbe, compute_rii,
                    generate_perturbations, compute_decision_scores,
                    rmse, BASE_STRATEGIES)
from utils.visualization import (plot_fig1, plot_fig2, plot_fig3, print_table3)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_RATIOS  = [0.05, 0.10, 0.20, 0.50, 1.00]
N_RUNS       = 20
N_PERT       = 50
INPUT_DIM    = 6          # [v, h, Q, T, P, D]
K_BASELINE   = 0.05       # Strategy A k value (baseline the model predicts)
V_IDX, D_IDX = 0, 5      # indices in feature array

STRATEGY_KS = [s.k for s in BASE_STRATEGIES]   # [0.05, 0.12, 0.25]

HPARAMS = {
    "PlainLSTM":      dict(hidden_dim=64, num_layers=2, dropout=0.2,
                           lr=0.001, max_epochs=500, patience=50),
    "FeatureAugLSTM": dict(hidden_dim=64, num_layers=2, dropout=0.2,
                           lr=0.001, max_epochs=500, patience=50),
    "StandardPINN":   dict(lr=0.0005, max_epochs=1000, patience=100),
    "PhyLSTM":        dict(hidden_dim=64, num_layers=2, dropout=0.2,
                           lambda_phy=0.1, lambda_bc=0.05,
                           lr=0.001, max_epochs=500, patience=50),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str):
    hp = HPARAMS[name]
    if name == "PlainLSTM":
        return PlainLSTM(INPUT_DIM, hp["hidden_dim"], hp["num_layers"],
                         dropout=hp["dropout"])
    if name == "FeatureAugLSTM":
        return FeatureAugLSTM(INPUT_DIM, hidden_dim=hp["hidden_dim"],
                              num_layers=hp["num_layers"], dropout=hp["dropout"])
    if name == "StandardPINN":
        return StandardPINN(INPUT_DIM, hidden_dims=[128, 64, 32])
    if name == "PhyLSTM":
        return PhyLSTM(INPUT_DIM, hp["hidden_dim"], hp["num_layers"],
                       dropout=hp["dropout"],
                       lambda_phy=hp["lambda_phy"], lambda_bc=hp["lambda_bc"])
    raise ValueError(name)


def build_trainer(model, name: str) -> Trainer:
    hp = HPARAMS[name]
    return Trainer(model, lr=hp["lr"], max_epochs=hp["max_epochs"],
                   patience=hp["patience"], k_decay=K_BASELINE)


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def single_run(dataset: dict, data_ratio: float, seed: int) -> dict:
    rng      = np.random.default_rng(seed)
    features = dataset["features"]   # (8, 120, 6)
    targets  = dataset["targets"]    # (8, 120)
    sC_all   = dataset["strategy_C"] # (8, 120, 3)

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = get_train_val_test_split(
        features, targets)
    T_val_end = int(N_TIMESTEPS * 0.8)
    sC_te     = sC_all[:, T_val_end:, :]    # (8, T_test, 3)

    X_sub, y_sub         = subsample_training_data(X_tr, y_tr, data_ratio, rng)
    X_sub_n, mu, sigma   = normalize(X_sub)
    X_v_n,   _, _        = normalize(X_v,  mu, sigma)
    X_te_n,  _, _        = normalize(X_te, mu, sigma)

    v_sub = X_sub_n[:, :, V_IDX];  D_sub = X_sub_n[:, :, D_IDX]
    v_v   = X_v_n[:,  :, V_IDX];   D_v   = X_v_n[:,  :, D_IDX]

    train_ds = make_dataset(X_sub_n, y_sub, v_sub, D_sub)
    val_ds   = make_dataset(X_v_n,   y_v,   v_v,   D_v)

    costs = [s.cost_dict() for s in BASE_STRATEGIES]
    C0    = float(targets.max())

    results = {}
    for name in ["PlainLSTM", "FeatureAugLSTM", "StandardPINN", "PhyLSTM"]:
        model   = build_model(name)
        trainer = build_trainer(model, name)
        trainer.fit(train_ds, val_ds)

        y_pred = trainer.predict(X_te_n)   # (8, T_test)
        r      = rmse(y_te, y_pred)

        rssi_per_sec  = []
        rssi_t_per_sec = []
        mbe_per_sec   = []

        for sec in range(N_SECTIONS):
            # ── CORRECT RSSI: propagate model error into strategy concentrations ──
            pred_error = y_pred[sec] - y_te[sec]   # (T_test,)  model-specific error

            # Nominal: ground-truth concentrations adjusted for this model's bias
            pred_sC_nom = propagate_prediction_error(
                sC_te[sec], pred_error, STRATEGY_KS, K_BASELINE)   # (T_test, 3)

            # Perturbed: add 5% noise to already-error-shifted concentrations
            pred_sC_pert = generate_perturbations(
                pred_sC_nom, n_pert=N_PERT, noise_pct=0.15, rng=rng)  # (N_PERT, T_test, 3)

            D_nom  = compute_decision_scores(pred_sC_nom,  costs, C0=C0)
            D_pert = np.stack([compute_decision_scores(pred_sC_pert[k], costs, C0=C0)
                               for k in range(N_PERT)])

            rssi_t = compute_rssi(D_nom, D_pert)    # (T_test,)
            rssi_per_sec.append(float(np.mean(rssi_t)))
            rssi_t_per_sec.append(rssi_t)

            # MBE: uses raw (un-normalised) v and D from test set
            v_sec = X_te[sec, :, V_IDX]
            D_sec = X_te[sec, :, D_IDX]
            mbe   = compute_mbe(y_pred[sec], v_sec, D_sec, K_BASELINE)
            mbe_per_sec.append(mbe)

        rssi_t_arr = np.array(rssi_t_per_sec)   # (8, T_test)
        mbe_arr    = np.array(mbe_per_sec)       # (8, T_test-1)

        results[name] = {
            "rmse":   r,
            "rssi":   float(np.mean(rssi_per_sec)),
            "rssi_t": rssi_t_arr.mean(axis=0),
            "mbe":    mbe_arr.mean(axis=0),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_data_scarcity_experiment(n_runs: int = N_RUNS,
                                  save_figures: bool = True) -> dict:
    print("=" * 60)
    print("Experiment: Data Scarcity Analysis")
    print(f"  Runs per configuration: {n_runs}")
    print(f"  Data ratios: {DATA_RATIOS}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    dataset = build_dataset(seed=42)

    model_names = ["PlainLSTM", "FeatureAugLSTM", "StandardPINN", "PhyLSTM"]
    labels = {
        "PlainLSTM":      "Plain LSTM",
        "FeatureAugLSTM": "Feature-augmented LSTM",
        "StandardPINN":   "Standard PINN",
        "PhyLSTM":        "PhyLSTM (proposed)",
    }

    all_results = {r: {n: [] for n in model_names} for r in DATA_RATIOS}

    for ratio in DATA_RATIOS:
        print(f"\n--- Training data: {ratio*100:.0f}% ---")
        for run in range(n_runs):
            if run % 5 == 0:
                print(f"  run {run+1}/{n_runs} …")
            res = single_run(dataset, ratio, seed=run)
            for name in model_names:
                all_results[ratio][name].append(res[name])

    # ── Aggregate ─────────────────────────────────────────────────────────
    table_data   = {labels[n]: {} for n in model_names}
    rmse_dict    = {labels[n]: [] for n in model_names}
    rssi_dict    = {labels[n]: [] for n in model_names}
    rmse_ci      = {labels[n]: [] for n in model_names}
    rssi_ci      = {labels[n]: [] for n in model_names}
    rssi_t_10    = {}
    rssi_ci_10   = {}
    mbe_100      = {}
    rii_runs_100 = {}

    for ratio in DATA_RATIOS:
        for name in model_names:
            runs  = all_results[ratio][name]
            lbl   = labels[name]
            rmses = np.array([r["rmse"] for r in runs])
            rssis = np.array([r["rssi"] for r in runs])

            rmse_dict[lbl].append(rmses.mean())
            rssi_dict[lbl].append(rssis.mean())
            rmse_ci[lbl].append(1.96 * rmses.std() / np.sqrt(len(rmses)))
            rssi_ci[lbl].append(1.96 * rssis.std() / np.sqrt(len(rssis)))

            key = f"{int(ratio*100)}%"
            table_data[lbl][key + "_rmse"] = (rmses.mean(), rmses.std())
            table_data[lbl][key + "_rssi"] = (rssis.mean(), rssis.std())

            if ratio == 0.10:
                rssi_t_arr        = np.array([r["rssi_t"] for r in runs])
                rssi_t_10[lbl]   = rssi_t_arr.mean(axis=0)
                rssi_ci_10[lbl]  = 1.96 * rssi_t_arr.std(axis=0) / np.sqrt(len(runs))

            if ratio == 1.00:
                mbe_arr          = np.array([r["mbe"] for r in runs])
                mbe_100[lbl]     = mbe_arr.mean(axis=0)
                # RII for box plots: per-run scalar
                rii_runs_100[lbl] = 1.0 - rssis   # (n_runs,)

    # ── Table 3 ───────────────────────────────────────────────────────────
    table_slim = {lbl: {k: v for k, v in d.items()
                         if k.startswith(("10%", "50%", "100%"))}
                  for lbl, d in table_data.items()}
    print_table3(table_slim)

    # ── Figures ───────────────────────────────────────────────────────────
    if save_figures:
        plot_fig1(DATA_RATIOS, rmse_dict, rssi_dict, rmse_ci, rssi_ci)
        plot_fig2(rssi_t_10, rssi_ci_10, mbe_100,
                  collapse_threshold=0.15, mbe_threshold=0.030, mbe_cross_month=60)
        sig_pairs = [
            ("Feature-augmented LSTM", "Plain LSTM", 0.05,  "dashed"),
            ("PhyLSTM (proposed)",     "Plain LSTM", 0.001, "solid"),
        ]
        plot_fig3(rii_runs_100, sig_pairs=sig_pairs)

    return {
        "all_results":   all_results,
        "table_data":    table_slim,
        "rssi_t_10":     rssi_t_10,
        "mbe_100":       mbe_100,
        "rii_runs_100":  rii_runs_100,
    }


if __name__ == "__main__":
    run_data_scarcity_experiment(n_runs=5, save_figures=True)
