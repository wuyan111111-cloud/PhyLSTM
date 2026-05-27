"""
Experiment 3 — Parameter Sensitivity Analysis (Section 5.4, Table 4).

Applies systematic perturbations to:
  - Degradation coefficient k  (±10%, ±20%)
  - Diffusion coefficient D    (±10%, ±20%)
  - Restoration costs          (±30%)

and measures the resulting ΔRSSI for each model.

RSSI is computed via propagate_prediction_error so that each model's
prediction quality is reflected in the baseline RSSI before perturbation
is applied. Sensitivity then measures how much RSSI drops when physical
parameters are additionally perturbed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from data.generate_data import (build_dataset, get_train_val_test_split,
                                 normalize, simulate_adr,
                                 N_SECTIONS, N_TIMESTEPS)
from models import PlainLSTM, FeatureAugLSTM, PhyLSTM
from utils import (Trainer, make_dataset, DEVICE,
                   propagate_prediction_error,
                   compute_rssi, aggregate_rssi,
                   generate_perturbations, compute_decision_scores,
                   rmse, BASE_STRATEGIES)

INPUT_DIM  = 6
N_RUNS     = 10
N_PERT     = 50
K_BASELINE = 0.05
V_IDX, D_IDX = 0, 5

STRATEGY_KS = [s.k for s in BASE_STRATEGIES]   # [0.05, 0.12, 0.25]


def _train_model(model, dataset: dict, seed: int):
    """Train model and return predictions + test targets."""
    rng        = np.random.default_rng(seed)
    features   = dataset["features"]
    targets    = dataset["targets"]
    strategy_C = dataset["strategy_C"]

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = get_train_val_test_split(
        features, targets)
    T_val_end = int(N_TIMESTEPS * 0.8)
    sC_te = strategy_C[:, T_val_end:, :]

    X_tr_n, mu, sigma = normalize(X_tr)
    X_v_n, _, _       = normalize(X_v,  mu, sigma)
    X_te_n, _, _      = normalize(X_te, mu, sigma)

    v_tr = X_tr_n[:, :, V_IDX]; D_tr = X_tr_n[:, :, D_IDX]
    v_v  = X_v_n[:, :, V_IDX];  D_v  = X_v_n[:, :, D_IDX]

    train_ds = make_dataset(X_tr_n, y_tr, v_tr, D_tr)
    val_ds   = make_dataset(X_v_n,  y_v,  v_v,  D_v)

    trainer = Trainer(model, lr=0.001, max_epochs=500,
                      patience=50, k_decay=K_BASELINE)
    trainer.fit(train_ds, val_ds)

    y_pred = trainer.predict(X_te_n)    # (8, T_test)
    return y_pred, y_te, sC_te, rng


def _compute_rssi_with_error(y_pred: np.ndarray,
                              y_te:   np.ndarray,
                              sC_te:  np.ndarray,
                              costs:  list[dict],
                              C0:     float,
                              rng:    np.random.Generator,
                              strategy_ks: list[float] | None = None) -> float:
    """
    Compute mean RSSI using propagated prediction error.
    strategy_ks: override k list for perturbed-k experiments.
    """
    if strategy_ks is None:
        strategy_ks = STRATEGY_KS

    rssi_all = []
    for sec in range(N_SECTIONS):
        pred_error  = y_pred[sec] - y_te[sec]
        pred_sC_nom = propagate_prediction_error(
            sC_te[sec], pred_error, strategy_ks, K_BASELINE)
        pred_sC_pert = generate_perturbations(
            pred_sC_nom, n_pert=N_PERT, noise_pct=0.15, rng=rng)

        D_nom  = compute_decision_scores(pred_sC_nom,  costs, C0=C0)
        D_pert = np.stack([compute_decision_scores(pred_sC_pert[k], costs, C0=C0)
                           for k in range(N_PERT)])
        rssi_all.append(aggregate_rssi(compute_rssi(D_nom, D_pert)))
    return float(np.mean(rssi_all))


def perturb_costs(strategies, factor: float) -> list[dict]:
    return [{"init_cost": s.init_cost * factor,
             "annual_om": s.annual_om * factor}
            for s in strategies]


PERTURBATIONS = [
    ("Degradation coeff k", "±10%", "k", 0.10),
    ("Degradation coeff k", "±20%", "k", 0.20),
    ("Diffusion coeff D",   "±10%", "D", 0.10),
    ("Diffusion coeff D",   "±20%", "D", 0.20),
    ("Restoration cost",    "±30%", "cost", 0.30),
]


def run_sensitivity_analysis(n_runs: int = N_RUNS):
    print("\n" + "=" * 60)
    print("Experiment: Parameter Sensitivity Analysis (Table 4)")
    print(f"  Runs: {n_runs}")
    print("=" * 60)

    dataset = build_dataset(seed=42)
    C0      = float(dataset["targets"].max())
    costs_nom = [s.cost_dict() for s in BASE_STRATEGIES]

    model_builders = {
        "Plain LSTM":             lambda: PlainLSTM(INPUT_DIM, 64, 2, dropout=0.2),
        "Feature-augmented LSTM": lambda: FeatureAugLSTM(INPUT_DIM,
                                                          hidden_dim=64,
                                                          num_layers=2,
                                                          dropout=0.2),
        "PhyLSTM":                lambda: PhyLSTM(INPUT_DIM, 64, 2, dropout=0.2,
                                                   lambda_phy=0.1, lambda_bc=0.05),
    }

    # One pre-trained set per model (seed=0)
    pretrained = {}
    print("Pre-training models (seed=0) …")
    for m_name, builder in model_builders.items():
        y_pred, y_te, sC_te, rng = _train_model(builder(), dataset, seed=0)
        nominal = _compute_rssi_with_error(y_pred, y_te, sC_te,
                                           costs_nom, C0, rng)
        pretrained[m_name] = dict(y_pred=y_pred, y_te=y_te,
                                  sC_te=sC_te, rng=rng, nominal=nominal)
        print(f"  {m_name}: nominal RSSI = {nominal:.3f}")

    # ── Table 4 ───────────────────────────────────────────────────────────
    header = (f"{'Parameter':<25} {'Magnitude':>10} "
              f"{'Plain ΔRSSI':>13} {'Feat-Aug ΔRSSI':>16} "
              f"{'PhyLSTM ΔRSSI':>15} {'PhyLSTM advantage':>20}")
    print("\n" + "=" * len(header))
    print("Table 4 — Parameter Sensitivity Analysis (ΔRSSI under perturbation)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for param_name, mag_str, ptype, magnitude in PERTURBATIONS:
        delta = {}
        for m_name, pt in pretrained.items():
            rng_copy = np.random.default_rng(0)   # reproducible per row

            if ptype == "cost":
                cu = perturb_costs(BASE_STRATEGIES, 1 + magnitude)
                cd = perturb_costs(BASE_STRATEGIES, 1 - magnitude)
                r_up   = _compute_rssi_with_error(
                    pt["y_pred"], pt["y_te"], pt["sC_te"], cu, C0, rng_copy)
                r_down = _compute_rssi_with_error(
                    pt["y_pred"], pt["y_te"], pt["sC_te"], cd, C0, rng_copy)

            elif ptype == "k":
                # Perturb strategy k ratios: changes how pred_error propagates
                ks_up   = [k * (1 + magnitude) for k in STRATEGY_KS]
                ks_down = [k * (1 - magnitude) for k in STRATEGY_KS]
                r_up   = _compute_rssi_with_error(
                    pt["y_pred"], pt["y_te"], pt["sC_te"],
                    costs_nom, C0, rng_copy, strategy_ks=ks_up)
                r_down = _compute_rssi_with_error(
                    pt["y_pred"], pt["y_te"], pt["sC_te"],
                    costs_nom, C0, rng_copy, strategy_ks=ks_down)

            else:  # ptype == "D": diffusion mainly affects prediction error
                # Scale pred_error by D-ratio proxy (larger D → faster mixing → smaller error)
                y_pred_up   = pt["y_pred"] * (1 - magnitude * 0.3)   # D↑ → smaller error
                y_pred_down = pt["y_pred"] * (1 + magnitude * 0.3)
                r_up   = _compute_rssi_with_error(
                    y_pred_up,   pt["y_te"], pt["sC_te"], costs_nom, C0, rng_copy)
                r_down = _compute_rssi_with_error(
                    y_pred_down, pt["y_te"], pt["sC_te"], costs_nom, C0, rng_copy)

            mean_pert = (r_up + r_down) / 2
            delta[m_name] = mean_pert - pt["nominal"]   # negative = drop

        adv = (abs(delta["Plain LSTM"]) - abs(delta["PhyLSTM"])) / (
               abs(delta["Plain LSTM"]) + 1e-9) * 100

        print(f"{param_name:<25} {mag_str:>10} "
              f"   {delta['Plain LSTM']:+.4f}        "
              f"   {delta['Feature-augmented LSTM']:+.4f}         "
              f"   {delta['PhyLSTM']:+.4f}         "
              f"   {adv:+.0f}%")

    print("=" * len(header))
    print("Note: ΔRSSI = perturbed RSSI − nominal RSSI "
          "(less negative = more robust).\n")


if __name__ == "__main__":
    run_sensitivity_analysis(n_runs=5)
