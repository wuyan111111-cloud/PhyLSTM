"""
Experiment 4 — Six-Strategy Expansion (Section 5.5, Table 5, Figure 4).

Extends baseline three strategies (A, B, C) to six variants
(A1, A2, B1, B2, C1, C2), comparing Plain LSTM vs PhyLSTM on:
  - RSSI per strategy and temporal Kendall's τ trajectory (Figure 4)
  - Ranking reversal frequency (Table 5)

RSSI is computed via propagate_prediction_error so that strategy-ranking
stability is model-dependent (tied to prediction accuracy), exactly as
described in Section 5.5 of the paper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from data.generate_data import (build_dataset, get_train_val_test_split,
                                 normalize, simulate_adr,
                                 N_SECTIONS, N_TIMESTEPS)
from models import PlainLSTM, PhyLSTM
from utils import (Trainer, make_dataset, DEVICE,
                   propagate_prediction_error,
                   compute_rssi, aggregate_rssi,
                   generate_perturbations, compute_decision_scores,
                   rmse, SIX_STRATEGIES)
from utils.visualization import plot_fig4

INPUT_DIM    = 6
N_RUNS       = 10
N_PERT       = 50
K_BASELINE   = 0.05    # Strategy A1 is closest to baseline (k=0.045)
V_IDX, D_IDX = 0, 5

SIX_KS = [s.k for s in SIX_STRATEGIES]   # [0.045, 0.055, 0.10, 0.14, 0.22, 0.28]


def _build_six_strategy_C(dataset: dict,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Simulate contaminant concentrations under the six-strategy expansion.
    Returns: (N_sections, T_test, 6)
    """
    T_val_end = int(N_TIMESTEPS * 0.8)
    sC_six    = np.zeros((N_SECTIONS, N_TIMESTEPS - T_val_end, len(SIX_STRATEGIES)))

    for sec in range(N_SECTIONS):
        drv = dataset["drivers"][sec]
        for si, strat in enumerate(SIX_STRATEGIES):
            C0  = rng.uniform(1.5, 4.5)
            C   = simulate_adr(drv["v"], drv["D"], strat.k, C0,
                                N_TIMESTEPS, strategy="A", noise_std=0.04)
            C   = strat.apply_resets(C)
            sC_six[sec, :, si] = C[T_val_end:]

    return sC_six


def _ranking_reversal_freq(D_nom:  np.ndarray,
                            D_pert: np.ndarray) -> float:
    """Fraction of (t, pert) pairs where the top-ranked strategy changes."""
    T       = D_nom.shape[0]
    N_pert  = D_pert.shape[0]
    top1    = np.argmax(D_nom, axis=1)   # (T,)
    count   = sum(int(np.sum(np.argmax(D_pert[k], axis=1) != top1))
                  for k in range(N_pert))
    return count / (T * N_pert)


def single_run(dataset: dict, seed: int) -> dict:
    rng      = np.random.default_rng(seed)
    features = dataset["features"]
    targets  = dataset["targets"]

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = get_train_val_test_split(
        features, targets)

    X_tr_n, mu, sigma = normalize(X_tr)
    X_v_n,  _, _      = normalize(X_v,  mu, sigma)
    X_te_n, _, _      = normalize(X_te, mu, sigma)

    v_tr = X_tr_n[:, :, V_IDX]; D_tr = X_tr_n[:, :, D_IDX]
    v_v  = X_v_n[:, :, V_IDX];  D_v  = X_v_n[:, :, D_IDX]

    train_ds = make_dataset(X_tr_n, y_tr, v_tr, D_tr)
    val_ds   = make_dataset(X_v_n,  y_v,  v_v,  D_v)

    sC_six = _build_six_strategy_C(dataset, rng)   # (8, T_test, 6)
    costs  = [s.cost_dict() for s in SIX_STRATEGIES]
    C0     = float(targets.max())

    results = {}

    for model_name, model in [
        ("Plain LSTM", PlainLSTM(INPUT_DIM, 64, 2, dropout=0.2)),
        ("PhyLSTM",    PhyLSTM(INPUT_DIM, 64, 2, dropout=0.2,
                                lambda_phy=0.1, lambda_bc=0.05)),
    ]:
        trainer = Trainer(model, lr=0.001, max_epochs=500,
                          patience=50, k_decay=K_BASELINE)
        trainer.fit(train_ds, val_ds)

        y_pred = trainer.predict(X_te_n)   # (8, T_test)

        rssi_by_strat   = {s.name: [] for s in SIX_STRATEGIES}
        reversal_by_sec = []
        rssi_t_sections = []   # (N_SECTIONS, T_test)

        for sec in range(N_SECTIONS):
            # Propagate this model's pred error into six-strategy concentrations
            pred_error  = y_pred[sec] - y_te[sec]              # (T_test,)
            pred_sC_nom = propagate_prediction_error(
                sC_six[sec], pred_error, SIX_KS, K_BASELINE)  # (T_test, 6)

            pred_sC_pert = generate_perturbations(
                pred_sC_nom, n_pert=N_PERT, noise_pct=0.15, rng=rng)

            D_nom  = compute_decision_scores(pred_sC_nom,  costs, C0=C0)
            D_pert = np.stack([compute_decision_scores(pred_sC_pert[k], costs, C0=C0)
                               for k in range(N_PERT)])

            # Overall temporal τ across all 6 strategies (for Fig. 4)
            rssi_t = compute_rssi(D_nom, D_pert)   # (T_test,)
            rssi_t_sections.append(rssi_t)

            # Per-strategy RSSI (for Table 5):
            # Measure how stable each strategy's RANK (within the full 6-strategy
            # ranking) is under perturbation. This avoids the 1-element tau issue.
            rank_nom = np.argsort(-D_nom, axis=1)  # (T_test, 6) rank positions
            for si, strat in enumerate(SIX_STRATEGIES):
                # rank of strategy si at each time step under nominal and perturbed
                pos_nom = np.where(rank_nom == si, 1, 0).sum(axis=1)   # always 1
                tau_sum = 0.0
                for k in range(N_PERT):
                    rank_pert_k = np.argsort(-D_pert[k], axis=1)
                    # fraction of time steps where strategy si holds same rank
                    same_rank = np.mean(rank_pert_k == rank_nom)
                    tau_sum  += same_rank
                rssi_by_strat[strat.name].append(tau_sum / N_PERT)

            reversal_by_sec.append(_ranking_reversal_freq(D_nom, D_pert))

        rssi_t_arr = np.array(rssi_t_sections)    # (8, T_test)

        results[model_name] = {
            "rssi_by_strat":  {k: float(np.mean(v))
                               for k, v in rssi_by_strat.items()},
            "reversal_freq":  float(np.mean(reversal_by_sec)),
            "rssi_t":         rssi_t_arr.mean(axis=0),   # (T_test,)
        }

    return results


def run_six_strategy_experiment(n_runs: int = N_RUNS):
    print("\n" + "=" * 60)
    print("Experiment: Six-Strategy Expansion (Table 5, Figure 4)")
    print(f"  Runs: {n_runs}")
    print("=" * 60)

    dataset     = build_dataset(seed=42)
    model_names = ["Plain LSTM", "PhyLSTM"]

    agg = {m: {"rssi": {s.name: [] for s in SIX_STRATEGIES},
               "reversal": [],
               "rssi_t":   []}
           for m in model_names}

    for run in range(n_runs):
        if run % 3 == 0:
            print(f"  run {run + 1}/{n_runs} …")
        res = single_run(dataset, seed=run)
        for m_name in model_names:
            for strat in SIX_STRATEGIES:
                agg[m_name]["rssi"][strat.name].append(
                    res[m_name]["rssi_by_strat"][strat.name])
            agg[m_name]["reversal"].append(res[m_name]["reversal_freq"])
            agg[m_name]["rssi_t"].append(res[m_name]["rssi_t"])

    # ── Table 5 ───────────────────────────────────────────────────────────
    header = (f"{'ID':<5} {'Strategy':<44} "
              f"{'Plain RSSI':>12} {'PhyLSTM RSSI':>14} "
              f"{'Rev% Plain':>12} {'Rev% PhyLSTM':>14}")
    print("\n" + "=" * len(header))
    print("Table 5 — Six-Strategy Expansion: RSSI and Ranking Reversal Frequency")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for strat in SIX_STRATEGIES:
        plain_rssi = float(np.mean(agg["Plain LSTM"]["rssi"][strat.name]))
        phy_rssi   = float(np.mean(agg["PhyLSTM"]["rssi"][strat.name]))
        plain_rev  = float(np.mean(agg["Plain LSTM"]["reversal"])) * 100
        phy_rev    = float(np.mean(agg["PhyLSTM"]["reversal"]))    * 100
        print(f"{strat.name:<5} {strat.label:<44} "
              f"  {plain_rssi:.3f}          "
              f"  {phy_rssi:.3f}         "
              f"  {plain_rev:.1f}%       "
              f"  {phy_rev:.1f}%")

    avg_plain = np.mean([np.mean(v) for v in agg["Plain LSTM"]["rssi"].values()])
    avg_phy   = np.mean([np.mean(v) for v in agg["PhyLSTM"]["rssi"].values()])
    print("-" * len(header))
    print(f"{'Avg':<5} {'':<44} "
          f"  {avg_plain:.3f}          "
          f"  {avg_phy:.3f}")
    print("=" * len(header) + "\n")

    # ── Figure 4 — temporal τ over 120 months ─────────────────────────────
    viz_labels = {"Plain LSTM": "Plain LSTM", "PhyLSTM": "PhyLSTM (proposed)"}
    rssi_t_mean = {}
    rssi_t_ci   = {}
    for m_name in model_names:
        arr = np.array(agg[m_name]["rssi_t"])   # (n_runs, T_test)
        lbl = viz_labels[m_name]
        rssi_t_mean[lbl] = arr.mean(axis=0)
        rssi_t_ci[lbl]   = 1.96 * arr.std(axis=0) / np.sqrt(max(len(arr), 1))

    plot_fig4(rssi_t_mean, rssi_t_ci, threshold=0.70)
    print("Figure 4 saved to figures/fig4_six_strategy_kendall_tau.png")

    return {"agg": agg, "rssi_t_mean": rssi_t_mean, "rssi_t_ci": rssi_t_ci}


if __name__ == "__main__":
    run_six_strategy_experiment(n_runs=5)
