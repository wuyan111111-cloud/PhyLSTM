"""
Experiment 2 — Ablation Study (Section 6.2, Table S5).

Tests the contribution of:
  (1) L_physics (ADR residual loss)
  (2) L_boundary (boundary condition penalty)

Four configurations:
  - PhyLSTM Full            : L_data + λ_phy * L_phy + λ_bc * L_bc
  - Disable L_phy           : L_data + λ_bc * L_bc
  - Disable L_bc            : L_data + λ_phy * L_phy
  - Disable All (Pure LSTM) : L_data only

RSSI computation uses propagate_prediction_error so that results are
model-dependent (configurations with stronger physics constraints produce
smaller prediction errors → smaller strategy-concentration distortion
→ higher, more stable RSSI).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from data.generate_data import (build_dataset, get_train_val_test_split,
                                 normalize, N_SECTIONS, N_TIMESTEPS)
from models.phylstm import PhyLSTM
from models.pinn    import boundary_loss, adr_residual
from utils import (Trainer, make_dataset, DEVICE,
                   propagate_prediction_error,
                   compute_rssi, aggregate_rssi, compute_mbe,
                   generate_perturbations, compute_decision_scores,
                   rmse, BASE_STRATEGIES)

N_RUNS     = 20
INPUT_DIM  = 6
K_BASELINE = 0.05        # Strategy A k (baseline model predicts)
N_PERT     = 50
V_IDX, D_IDX = 0, 5

STRATEGY_KS = [s.k for s in BASE_STRATEGIES]   # [0.05, 0.12, 0.25]


class AblatedPhyLSTM(PhyLSTM):
    """PhyLSTM with selective component disabling."""

    def __init__(self, *args, use_phy: bool = True, use_bc: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_phy = use_phy
        self.use_bc  = use_bc

    def compute_loss(self, x, y_true, v, D, k, C_bc=None):
        y_pred_raw, _ = self.forward(x)
        y_pred = y_pred_raw.squeeze(-1)

        L_data = ((y_pred - y_true) ** 2).mean()

        if self.use_phy and y_pred.shape[1] >= 3:
            r     = adr_residual(y_pred, v, D, k, self.dx, self.dt)
            L_phy = (r ** 2).mean()
        else:
            L_phy = torch.tensor(0.0, device=x.device)

        if self.use_bc and C_bc is not None:
            L_bc = boundary_loss(y_pred, C_bc)
        else:
            L_bc = torch.tensor(0.0, device=x.device)

        total = (L_data
                 + (self.lambda_phy * L_phy if self.use_phy else 0)
                 + (self.lambda_bc  * L_bc  if self.use_bc  else 0))
        return {"total": total, "data": L_data,
                "physics": L_phy, "boundary": L_bc}


CONFIGS = {
    "PhyLSTM (Full)":        dict(use_phy=True,  use_bc=True),
    "Disable L_phy":         dict(use_phy=False, use_bc=True),
    "Disable L_bc":          dict(use_phy=True,  use_bc=False),
    "Disable All (Pure LSTM)": dict(use_phy=False, use_bc=False),
}


def single_ablation_run(dataset: dict, seed: int) -> dict:
    rng        = np.random.default_rng(seed)
    features   = dataset["features"]
    targets    = dataset["targets"]
    strategy_C = dataset["strategy_C"]

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = get_train_val_test_split(
        features, targets)
    T_val_end = int(N_TIMESTEPS * 0.8)
    sC_te = strategy_C[:, T_val_end:, :]    # (8, T_test, 3)

    X_tr_n, mu, sigma = normalize(X_tr)
    X_v_n, _, _       = normalize(X_v,  mu, sigma)
    X_te_n, _, _      = normalize(X_te, mu, sigma)

    v_tr = X_tr_n[:, :, V_IDX];  D_tr = X_tr_n[:, :, D_IDX]
    v_v  = X_v_n[:, :, V_IDX];   D_v  = X_v_n[:, :, D_IDX]

    train_ds = make_dataset(X_tr_n, y_tr, v_tr, D_tr)
    val_ds   = make_dataset(X_v_n,  y_v,  v_v,  D_v)
    costs    = [s.cost_dict() for s in BASE_STRATEGIES]
    C0       = float(targets.max())

    results = {}

    for cfg_name, flags in CONFIGS.items():
        model = AblatedPhyLSTM(
            INPUT_DIM, hidden_dim=64, num_layers=2,
            dropout=0.2, lambda_phy=0.1, lambda_bc=0.05,
            **flags
        ).to(DEVICE)

        trainer = Trainer(model, lr=0.001, max_epochs=500,
                          patience=50, k_decay=K_BASELINE)
        trainer.fit(train_ds, val_ds)

        y_pred = trainer.predict(X_te_n)    # (8, T_test)
        r      = rmse(y_te, y_pred)

        rssi_all  = []
        mbe_final = []

        for sec in range(N_SECTIONS):
            # Propagate this configuration's prediction error into strategy concentrations
            pred_error  = y_pred[sec] - y_te[sec]              # (T_test,)
            pred_sC_nom = propagate_prediction_error(
                sC_te[sec], pred_error, STRATEGY_KS, K_BASELINE)  # (T_test, 3)

            pred_sC_pert = generate_perturbations(
                pred_sC_nom, n_pert=N_PERT, noise_pct=0.15, rng=rng)

            D_nom  = compute_decision_scores(pred_sC_nom,  costs, C0=C0)
            D_pert = np.stack([compute_decision_scores(pred_sC_pert[k], costs, C0=C0)
                               for k in range(N_PERT)])
            rssi_all.append(aggregate_rssi(compute_rssi(D_nom, D_pert)))

            v_sec = X_te[sec, :, V_IDX]
            D_sec = X_te[sec, :, D_IDX]
            mbe   = compute_mbe(y_pred[sec], v_sec, D_sec, K_BASELINE)
            mbe_final.append(mbe[-1] if len(mbe) > 0 else 0.0)

        mean_rssi = float(np.mean(rssi_all))
        results[cfg_name] = {
            "rmse":      r,
            "rssi":      mean_rssi,
            "rii":       1.0 - mean_rssi,
            "mbe_final": float(np.mean(mbe_final)),
        }

    return results


def run_ablation_study(n_runs: int = N_RUNS):
    print("\n" + "=" * 60)
    print("Experiment: Ablation Study (Table S5)")
    print(f"  Runs: {n_runs}")
    print("=" * 60)

    dataset  = build_dataset(seed=42)
    all_runs = {c: [] for c in CONFIGS}

    for run in range(n_runs):
        if run % 5 == 0:
            print(f"  run {run + 1}/{n_runs} …")
        res = single_ablation_run(dataset, seed=run)
        for cfg, vals in res.items():
            all_runs[cfg].append(vals)

    ref_rii  = float(np.mean([r["rii"]  for r in all_runs["PhyLSTM (Full)"]]))
    ref_rmse = float(np.mean([r["rmse"] for r in all_runs["PhyLSTM (Full)"]]))

    header = (f"{'Configuration':<30} {'L_phy':^6} {'L_bc':^6} "
              f"{'RMSE':>10} {'RII':>10} {'MBE(final)':>12} "
              f"{'ΔRII':>10} {'ΔRMSE':>10}")
    print("\n" + "=" * len(header))
    print("Table S5 — Ablation Study")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cfg_name, flags in CONFIGS.items():
        runs   = all_runs[cfg_name]
        r_mean = float(np.mean([r["rmse"]      for r in runs]))
        r_std  = float(np.std( [r["rmse"]      for r in runs]))
        i_mean = float(np.mean([r["rii"]       for r in runs]))
        i_std  = float(np.std( [r["rii"]       for r in runs]))
        m_mean = float(np.mean([r["mbe_final"] for r in runs]))

        d_rii  = (i_mean - ref_rii)  / (ref_rii  + 1e-9) * 100
        d_rmse = (r_mean - ref_rmse) / (ref_rmse + 1e-9) * 100

        phy_mark = "✓" if flags.get("use_phy", True) else "✗"
        bc_mark  = "✓" if flags.get("use_bc",  True) else "✗"

        print(f"{cfg_name:<30} {phy_mark:^6} {bc_mark:^6} "
              f" {r_mean:.3f}±{r_std:.3f} "
              f" {i_mean:.3f}±{i_std:.3f} "
              f"   {m_mean:.4f}   "
              f"  {d_rii:+.0f}%    "
              f"  {d_rmse:+.1f}%")

    print("=" * len(header) + "\n")
    return all_runs


if __name__ == "__main__":
    run_ablation_study(n_runs=5)
