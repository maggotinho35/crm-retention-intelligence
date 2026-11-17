"""Phase 3: tool and architecture selection, with a robustness test.

Reproduces the portfolio's weighted-scoring decision exactly (A 3.50, B 4.20, C 3.35) and then asks the
question a sceptical manager asks: "how sure are you? you picked the weights yourself."

  * Monte Carlo over the criteria weights (Dirichlet centred on the chosen weights),
  * Monte Carlo over the 1-5 scores (each score nudged by Gaussian noise, clipped to [1, 5]),
  * the weight on cost-efficiency at which the open-source stack would overtake the Microsoft stack.
"""
import numpy as np
import pandas as pd

from .config import SEED

CRITERIA = ["Cost-efficiency", "Functional coverage", "CRM integration", "Ease of use / skills", "Scalability"]
WEIGHTS = np.array([0.20, 0.30, 0.25, 0.10, 0.15])
SCENARIOS = {"A: Open-source stack": [5, 4, 2, 2, 4], "B: Microsoft stack": [3, 4, 5, 4, 5],
             "C: Enterprise suite (SAS/IBM)": [1, 5, 3, 3, 4]}
S = np.array(list(SCENARIOS.values()), dtype=float)
NAMES = list(SCENARIOS)


def weighted_scores():
    return pd.Series(S @ WEIGHTS, index=NAMES).round(2)


def monte_carlo(n=20000, concentration=25.0, score_sd=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(WEIGHTS * concentration, size=n)          # random weights near the chosen ones
    noisy = np.clip(S[None, :, :] + rng.normal(0, score_sd, (n, *S.shape)), 1, 5)
    totals_w = w @ S.T                                           # weights vary, scores fixed
    totals_both = np.einsum("nsc,nc->ns", noisy, w)              # weights and scores vary
    out = {}
    for label, t in [("weights vary", totals_w), ("weights and scores vary", totals_both)]:
        wins = np.bincount(t.argmax(1), minlength=len(NAMES)) / n
        out[label] = dict(zip(NAMES, wins))
    return pd.DataFrame(out).T


def cost_weight_breakeven():
    """Weight on cost at which A overtakes B when the other weights keep their relative proportions."""
    others = WEIGHTS.copy()
    others[0] = 0
    others = others / others.sum()
    sweep = np.linspace(0, 1, 1001)
    rows = []
    for wc in sweep:
        w = others * (1 - wc)
        w[0] = wc
        rows.append(S @ w)
    sc = pd.DataFrame(rows, index=sweep, columns=NAMES)
    flips = sc.index[sc[NAMES[0]] > sc[NAMES[1]]]
    return sc, (float(flips[0]) if len(flips) else float("nan"))
