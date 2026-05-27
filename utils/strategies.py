"""
Restoration strategy definitions (Table 2 / Table S2).

Three baseline strategies and their six-strategy expansion (Table 5).
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RestorationStrategy:
    """Container for a single restoration strategy."""
    name:          str
    label:         str
    intensity:     str
    k:             float           # degradation coefficient (d⁻¹)
    D:             float           # diffusion coefficient (m²/s)
    init_cost:     float           # initial investment (10⁴ CNY)
    annual_om:     float           # annual O&M cost    (10⁴ CNY)
    duration:      tuple[int, int] # implementation period (years)
    resets:        dict[int, float] = field(default_factory=dict)
    description:   str = ""

    def apply_resets(self, C: np.ndarray) -> np.ndarray:
        """Apply discrete state resets to a concentration trajectory."""
        C_out = C.copy()
        for t, alpha in self.resets.items():
            if t < len(C_out):
                C_out[t] = alpha * C_out[t - 1] if t > 0 else alpha * C_out[t]
                C_out[t] = max(C_out[t], 0.0)
        return C_out

    def cost_dict(self) -> dict:
        return {"init_cost": self.init_cost, "annual_om": self.annual_om}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline three strategies (Table 2)
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_A = RestorationStrategy(
    name      = "A",
    label     = "Natural Attenuation",
    intensity = "Low",
    k         = 0.05,
    D         = 0.008,
    init_cost = 0,
    annual_om = 0,
    duration  = (15, 20),
    resets    = {},
    description = "No artificial intervention; long-term monitoring.",
)

STRATEGY_B = RestorationStrategy(
    name      = "B",
    label     = "Phytoremediation",
    intensity = "Medium",
    k         = 0.12,
    D         = 0.010,
    init_cost = 120,
    annual_om = 15,
    duration  = (8, 12),
    resets    = {},
    description = "Vegetation cover; biological uptake.",
)

STRATEGY_C = RestorationStrategy(
    name      = "C",
    label     = "Active Replacement",
    intensity = "High",
    k         = 0.25,
    D         = 0.012,
    init_cost = 580,
    annual_om = 45,
    duration  = (3, 5),
    resets    = {30: 0.20, 60: 0.40, 90: 0.50},
    description = "Engineering intervention; discrete resets at t=30,60,90 months.",
)

BASE_STRATEGIES = [STRATEGY_A, STRATEGY_B, STRATEGY_C]

# ─────────────────────────────────────────────────────────────────────────────
# Six-strategy expansion (Table 5) — parameter variants
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_A1 = RestorationStrategy(
    name="A1", label="Natural Attenuation – Low Monitoring",
    intensity="Low", k=0.045, D=0.007,
    init_cost=0, annual_om=0, duration=(18, 22),
    description="Natural attenuation with low monitoring frequency.",
)
STRATEGY_A2 = RestorationStrategy(
    name="A2", label="Natural Attenuation – High Monitoring",
    intensity="Low", k=0.055, D=0.009,
    init_cost=10, annual_om=5, duration=(15, 20),
    description="Natural attenuation with high monitoring frequency.",
)
STRATEGY_B1 = RestorationStrategy(
    name="B1", label="Phytoremediation – Low Density",
    intensity="Medium", k=0.10, D=0.009,
    init_cost=90, annual_om=10, duration=(10, 14),
    description="Phytoremediation with low vegetation density.",
)
STRATEGY_B2 = RestorationStrategy(
    name="B2", label="Phytoremediation – High Density",
    intensity="Medium", k=0.14, D=0.011,
    init_cost=150, annual_om=20, duration=(7, 10),
    description="Phytoremediation with high vegetation density.",
)
STRATEGY_C1 = RestorationStrategy(
    name="C1", label="Active Replacement – Standard Intensity",
    intensity="High", k=0.22, D=0.011,
    init_cost=500, annual_om=40, duration=(4, 6),
    resets={30: 0.22, 60: 0.42, 90: 0.52},
    description="Standard active replacement intensity.",
)
STRATEGY_C2 = RestorationStrategy(
    name="C2", label="Active Replacement – Enhanced Intensity",
    intensity="High", k=0.28, D=0.013,
    init_cost=650, annual_om=50, duration=(2, 4),
    resets={30: 0.18, 60: 0.38, 90: 0.48},
    description="Enhanced active replacement intensity.",
)

SIX_STRATEGIES = [STRATEGY_A1, STRATEGY_A2,
                  STRATEGY_B1, STRATEGY_B2,
                  STRATEGY_C1, STRATEGY_C2]

# ─────────────────────────────────────────────────────────────────────────────
# Decision weight configurations (Table S6)
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_CONFIGS = {
    "Baseline":           np.array([0.30, 0.30, 0.25, 0.15]),
    "Efficiency-Oriented":np.array([0.50, 0.20, 0.20, 0.10]),
    "Cost-Oriented":      np.array([0.20, 0.50, 0.20, 0.10]),
    "Ecology-Oriented":   np.array([0.20, 0.20, 0.50, 0.10]),
    "Risk-Averse":        np.array([0.25, 0.25, 0.25, 0.25]),
    "Equal Weights":      np.array([0.25, 0.25, 0.25, 0.25]),
}
