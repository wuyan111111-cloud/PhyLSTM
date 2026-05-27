"""
Evaluation metrics used in the paper.

Main text metrics:
  - RMSE  : Root Mean Squared Error (pointwise prediction accuracy)
  - RSSI  : Ranking Stability Sensitivity Index = Kendall's τ (Section 2.2)

Supplementary metrics:
  - MBE   : Cumulative Mass Balance Error (S3.2)
  - RII   : Ranking Instability Index = 1 − τ (Table S5, S6)

Key function: propagate_prediction_error
  Converts model-specific prediction errors into strategy-specific
  concentration trajectories, so that RSSI is model-dependent and
  physically meaningful (see Section 2.2 and S3.1).
"""

import numpy as np
from scipy.stats import kendalltau
from typing import Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise accuracy
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Core RSSI fix: propagate prediction error into strategy concentrations
# ─────────────────────────────────────────────────────────────────────────────

def propagate_prediction_error(C_strategies_true: np.ndarray,
                                pred_error: np.ndarray,
                                strategy_ks: list[float],
                                k_baseline: float) -> np.ndarray:
    """
    Convert model prediction error into strategy-specific concentration
    trajectories. This is the key function that makes RSSI model-dependent.

    Physical rationale (Section 2.2 / S3.1):
      Each strategy has a different degradation coefficient k_i. When the model
      makes a prediction error ε_t on the baseline concentration, the error
      propagates differently to each strategy's concentration because of their
      different k ratios: strategy i shifts by ε_t * (k_i / k_baseline).
      Strategies with larger k (more active) are MORE sensitive to prediction
      errors, so their concentrations are shifted more, making rankings more
      volatile under inaccurate predictions.

    Parameters
    ----------
    C_strategies_true : (T, n_strategies)  ground-truth strategy concentrations
    pred_error        : (T,)               model prediction error = y_pred - y_true
                        for a SINGLE monitoring section (already temporally aligned)
    strategy_ks       : list of k values, one per strategy
    k_baseline        : k of the baseline trajectory the model was trained to predict

    Returns
    -------
    C_pred_strategies : (T, n_strategies)  model-implied strategy concentrations
    """
    T, n_strat = C_strategies_true.shape
    assert len(strategy_ks) == n_strat, \
        "strategy_ks must have one entry per strategy column"
    assert len(pred_error) == T, \
        f"pred_error length {len(pred_error)} != T={T}"

    C_out = C_strategies_true.copy()
    for i, ki in enumerate(strategy_ks):
        # Scale error by k ratio: more active strategies amplify the error more
        scale = ki / max(k_baseline, 1e-6)
        C_out[:, i] = C_out[:, i] + pred_error * scale
        C_out[:, i] = np.clip(C_out[:, i], 0.0, None)

    return C_out


# ─────────────────────────────────────────────────────────────────────────────
# Strategy scoring (Section 4.3)
# ─────────────────────────────────────────────────────────────────────────────

def remediation_efficiency(C: np.ndarray, C0: float,
                           C_target: float = 0.5) -> np.ndarray:
    """
    y_eff = (C0 - C_t) / (C0 - C_target), clipped to [0, 1].  (Section 4.3)
    """
    denom = C0 - C_target
    if abs(denom) < 1e-8:
        return np.zeros_like(C)
    eff = (C0 - C) / denom
    return np.clip(eff, 0, 1)


def lifecycle_cost_score(init_cost: float, annual_om: float,
                         T: int, discount_rate: float = 0.05,
                         all_PVs: Sequence[float] | None = None) -> float:
    """
    y_cost = 1 - (PV_i - min(PV)) / (max(PV) - min(PV))   (Section 4.3)

    If `all_PVs` is provided it is used for normalisation; otherwise returns
    the raw PV so the caller can normalise across strategies.
    """
    pv = init_cost + sum(annual_om / (1 + discount_rate) ** t
                         for t in range(1, T + 1))
    if all_PVs is None:
        return pv  # raw PV for later normalisation

    min_pv = min(all_PVs)
    max_pv = max(all_PVs)
    if abs(max_pv - min_pv) < 1e-8:
        return 1.0
    return 1.0 - (pv - min_pv) / (max_pv - min_pv)


def ecosystem_service_value(C: np.ndarray, C_ref: float = 3.0,
                             unit_value: float = 1.0) -> np.ndarray:
    """
    Approximate ecosystem service value via replacement cost method.
    Scales inversely with contamination level (Section 4.3 / S2.3).
    """
    esv = unit_value * np.maximum(0, (C_ref - C) / C_ref)
    return np.clip(esv, 0, 1)


def ecological_risk_score(C: np.ndarray, threshold: float = 3.0) -> float:
    """
    y_risk = 1 - (1/T) * Σ I(C_t > threshold)   (Section 4.3)
    """
    return float(1.0 - np.mean(C > threshold))


def compute_decision_scores(C_strategies: np.ndarray,
                             strategy_costs: list[dict],
                             weights: np.ndarray | None = None,
                             C0: float = 4.0,
                             C_target: float = 0.5,
                             T_horizon: int | None = None) -> np.ndarray:
    """
    Compute composite decision score D_t^(i) = w^T y_t^(i) for each strategy
    at each time step (Section 4.3).

    Parameters
    ----------
    C_strategies  : (T, n_strategies)  predicted concentrations per strategy
    strategy_costs: list of dicts with keys 'init_cost', 'annual_om'
    weights       : (4,) weight vector [w_eff, w_cost, w_eco, w_risk]
                    default = [0.30, 0.30, 0.25, 0.15] (Table S6 baseline)
    C0            : float  initial / reference concentration
    C_target      : float  remediation target
    T_horizon     : int    number of time steps used for cost discounting

    Returns
    -------
    D : (T, n_strategies)  composite decision scores
    """
    if weights is None:
        weights = np.array([0.30, 0.30, 0.25, 0.15])  # Table S6 baseline

    T, n_strat = C_strategies.shape
    if T_horizon is None:
        T_horizon = T

    pvs = [lifecycle_cost_score(sc["init_cost"], sc["annual_om"], T_horizon)
           for sc in strategy_costs]
    min_pv, max_pv = min(pvs), max(pvs)

    D = np.zeros((T, n_strat))

    for i, (Ci, sc) in enumerate(zip(C_strategies.T, strategy_costs)):
        y_eff  = remediation_efficiency(Ci, C0, C_target)
        pv     = pvs[i]
        y_cost = 1.0 - (pv - min_pv) / max(max_pv - min_pv, 1e-8)
        y_cost = np.full(T, y_cost)
        y_eco  = ecosystem_service_value(Ci)
        y_risk = np.full(T, ecological_risk_score(Ci))

        Y = np.stack([y_eff, y_cost, y_eco, y_risk], axis=1)  # (T, 4)
        D[:, i] = Y @ weights

    return D


# ─────────────────────────────────────────────────────────────────────────────
# RSSI — Ranking Stability Sensitivity Index (Section 2.2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rssi(D_nominal: np.ndarray,
                 D_perturbed: np.ndarray) -> np.ndarray:
    """
    RSSI_t = (1/N_pert) Σ_k τ(π_t^(0), π_t^(δ_k))   (Eq. in Section 2.2)

    Parameters
    ----------
    D_nominal   : (T, n_strat)         nominal decision scores
    D_perturbed : (N_pert, T, n_strat) perturbed decision scores

    Returns
    -------
    rssi : (T,)  per time-step RSSI values
    """
    T      = D_nominal.shape[0]
    N_pert = D_perturbed.shape[0]
    rssi   = np.zeros(T)

    for t in range(T):
        rank_nom = np.argsort(-D_nominal[t])
        tau_sum  = 0.0
        for k in range(N_pert):
            rank_pert = np.argsort(-D_perturbed[k, t])
            tau, _    = kendalltau(rank_nom, rank_pert)
            tau_sum  += max(tau, -1.0)
        rssi[t] = tau_sum / N_pert

    return rssi


def aggregate_rssi(rssi: np.ndarray) -> float:
    """Scalar summary: mean RSSI over the prediction horizon."""
    return float(np.mean(rssi))


def compute_rii(rssi: np.ndarray) -> np.ndarray:
    """
    Ranking Instability Index = 1 − τ  (Table S5, S6).
    Lower is more stable.
    """
    return 1.0 - rssi


# ─────────────────────────────────────────────────────────────────────────────
# Mass Balance Error (MBE) — Section S3.2
# ─────────────────────────────────────────────────────────────────────────────

def compute_mbe(C_pred: np.ndarray,
                v:      np.ndarray,
                D:      np.ndarray,
                k:      float,
                dx:     float = 100.0,
                dt:     float = 1.0) -> np.ndarray:
    """
    Cumulative mass balance error.

        MBE_t = Σ_{i=1}^{t} | (Ĉ_i - Ĉ_{i-1}) - Φ(Ĉ_i, θ) |

    where Φ is the ADR theoretical flux.

    Parameters
    ----------
    C_pred : (T,)  predicted concentrations (single trajectory)
    v      : (T,)  advection velocities
    D      : (T,)  diffusion coefficients
    k      : float  degradation coefficient

    Returns
    -------
    mbe : (T-1,) cumulative MBE
    """
    T   = len(C_pred)
    mbe = np.zeros(T - 1)
    cum = 0.0

    for t in range(1, T):
        C_t   = C_pred[t]
        C_tm1 = C_pred[t - 1]
        vt    = v[t - 1]

        adv  = vt * (C_t - C_tm1) / dx
        rxn  = k  * C_t
        flux = dt * (adv + rxn)

        obs_change = C_t - C_tm1
        cum       += abs(obs_change - flux)
        mbe[t - 1] = cum

    return mbe


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation generator for RSSI computation
# ─────────────────────────────────────────────────────────────────────────────

def generate_perturbations(C_strategies: np.ndarray,
                            n_pert:    int   = 50,
                            noise_pct: float = 0.05,
                            rng: np.random.Generator | None = None
                            ) -> np.ndarray:
    """
    Generate N_pert perturbed versions of C_strategies by adding
    Gaussian noise proportional to the signal magnitude (5% default).

    Parameters
    ----------
    C_strategies : (T, n_strat)
    n_pert       : number of perturbation scenarios
    noise_pct    : relative noise level (0.05 = 5%)

    Returns
    -------
    perturbed : (n_pert, T, n_strat)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    std = noise_pct * np.abs(C_strategies)
    perturbed = np.stack([
        np.clip(C_strategies + rng.normal(0, 1, C_strategies.shape) * std, 0, None)
        for _ in range(n_pert)
    ], axis=0)
    return perturbed
