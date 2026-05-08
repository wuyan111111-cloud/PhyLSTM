"""
main.py — Run all PhyLSTM experiments end-to-end.

Usage:
    python main.py                          # full paper replication (n_runs=20)
    python main.py --quick                  # quick smoke-test  (n_runs=3)
    python main.py --experiment data        # only data-scarcity experiment
    python main.py --experiment ablation    # only ablation study
    python main.py --experiment sensitivity # only sensitivity analysis
    python main.py --experiment six_strat   # only six-strategy expansion
    python main.py --experiment weight      # only weight robustness

Experiment → paper table / figure mapping
------------------------------------------
data        → Table 3, Table S4, Figures 2–5
ablation    → Table 4a (main text); verification calcs in SI S3.3
sensitivity → Table 4  (parameter sensitivity, §5.5)
six_strat   → Table 5  (six-strategy expansion, §5.5)
weight      → Table S7 (weight robustness, SI S3.4)
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def _banner(title: str):
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def main():
    parser = argparse.ArgumentParser(
        description="PhyLSTM — Physics-Guided Sequential Learning for "
                    "Ecological Restoration Decision Stability")
    parser.add_argument("--quick",      action="store_true",
                        help="Run with n_runs=3 for a quick smoke-test")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "data", "ablation",
                                 "sensitivity", "six_strat", "weight"],
                        help="Which experiment to run (default: all)")
    parser.add_argument("--n_runs",     type=int, default=None,
                        help="Override number of independent runs "
                             "(default: 3 for --quick, 20 otherwise)")
    args = parser.parse_args()

    n_runs = args.n_runs or (3 if args.quick else 20)

    t0 = time.time()

    _banner("PhyLSTM — Reproducing all paper experiments")
    print(f"  Experiment : {args.experiment}")
    print(f"  n_runs     : {n_runs}")
    print(f"  Physics residual: lumped temporal mass balance (no Δx required)")
    print(f"  N_pert (RSSI)   : 1 000  |  perturbation amplitude: ±10%")

    # ── Data scarcity (Table 3, Table S4, Figures 2–5) ────────────────────
    if args.experiment in ("all", "data"):
        _banner("Experiment 1 — Data Scarcity Analysis  [Table 3 / Table S4, Figs 2–5]")
        from experiments.data_scarcity import run_data_scarcity_experiment
        run_data_scarcity_experiment(n_runs=n_runs, save_figures=True)

    # ── Ablation study (Table 4a, main text) ──────────────────────────────
    #    Includes: PhyLSTM Full / Disable ℒ_phy / Disable ℒ_bc /
    #              Disable All (Pure LSTM) / L2-LSTM†
    #    SI S3.3 contains the verification calculations for this table.
    if args.experiment in ("all", "ablation"):
        _banner("Experiment 2 — Ablation Study  [Table 4a, main text]")
        from experiments.ablation import run_ablation_study
        run_ablation_study(n_runs=n_runs)

    # ── Parameter sensitivity (Table 4, §5.5) ─────────────────────────────
    if args.experiment in ("all", "sensitivity"):
        _banner("Experiment 3 — Parameter Sensitivity Analysis  [Table 4, §5.5]")
        from experiments.sensitivity import run_sensitivity_analysis
        run_sensitivity_analysis(n_runs=max(n_runs // 2, 3))

    # ── Six-strategy expansion (Table 5, §5.5) ────────────────────────────
    if args.experiment in ("all", "six_strat"):
        _banner("Experiment 4 — Six-Strategy Expansion  [Table 5, §5.5]")
        from experiments.six_strategy import run_six_strategy_experiment
        run_six_strategy_experiment(n_runs=n_runs)

    # ── Weight robustness (Table S7, SI S3.4) ─────────────────────────────
    if args.experiment in ("all", "weight"):
        _banner("Experiment 5 — Weight Robustness Analysis  [Table S7, SI S3.4]")
        from experiments.weight_robustness import run_weight_robustness
        run_weight_robustness(n_runs=n_runs)

    elapsed = time.time() - t0
    _banner(f"All done — total elapsed {elapsed / 60:.1f} min")
    print("  Figures saved to : ./figures/")
    print("  Tables printed above.")
    print()


if __name__ == "__main__":
    main()
