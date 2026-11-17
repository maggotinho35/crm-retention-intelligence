"""Translate model output into euros and decisions.

What is MEASURED here: how many real churners each targeting strategy reaches in the held-out customers.
What is ASSUMED: euro value per customer, cost per contact and the share of reached churners who are saved
(see config.BusinessAssumptions). Every euro figure is therefore a scenario, shown with a sensitivity grid.

Per-customer value protected if saved  = (base revenue + rate * average balance) * horizon_years
Realised net value of contacting i     = churned_i * save_rate * value_i - offer_cost
"""
import numpy as np
import pandas as pd

from .config import SEED, BusinessAssumptions


def customer_value(df: pd.DataFrame, a: BusinessAssumptions) -> np.ndarray:
    return (a.base_annual_revenue_eur + a.revenue_rate_on_balance * df["avg_balance"].values) * a.retention_horizon_years


def _evaluate(df, mask, a, label):
    v = customer_value(df, a)
    y = df["churned"].values
    contacted = int(mask.sum())
    reached = int((y[mask] == 1).sum())
    saved_value = float((y[mask] * a.save_rate * v[mask]).sum())
    cost = contacted * a.offer_cost_eur
    n = len(df)
    return {"strategy": label, "contacted": contacted, "churners_reached": reached,
            "precision": reached / contacted if contacted else np.nan, "recall": reached / max(y.sum(), 1),
            "cost_eur": cost, "value_protected_eur": saved_value, "net_value_eur": saved_value - cost,
            "net_per_1000_customers": (saved_value - cost) / n * 1000,
            "roi": (saved_value - cost) / cost if cost else np.nan}


def _top_k(score, k):
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score)[:k]] = True
    return mask


def value_gated_mask(df, a):
    """Recommended policy: contact the customers with the highest expected net value, but only if it is > 0,
    and never more than the team's capacity."""
    n = len(df)
    k = int(round(a.team_capacity_share * n))
    ev = df["p_churn"].values * a.save_rate * customer_value(df, a) - a.offer_cost_eur
    return _top_k(ev, k) & (ev > 0)


def rule_mask(df):
    """Portfolio decision rule: recent fee increase AND two or more complaints in 12 months."""
    return ((df["had_fee_increase"] == 1) & (df["complaint_count_12m"] >= 2)).values


def compare_strategies(df: pd.DataFrame, a: BusinessAssumptions) -> pd.DataFrame:
    n = len(df)
    k = int(round(a.team_capacity_share * n))
    v = customer_value(df, a)
    p = df["p_churn"].values
    rows = [_evaluate(df, np.zeros(n, bool), a, "No action")]
    base_all = _evaluate(df, np.ones(n, bool), a, f"Blanket campaign (all {n})")
    rows.append(base_all)
    # random targeting: exact expectation (each customer contacted with probability k/n)
    exp = {key: (val * k / n if key in ("contacted", "churners_reached", "cost_eur", "value_protected_eur") else None)
           for key, val in base_all.items() if key != "strategy"}
    exp["net_value_eur"] = exp["value_protected_eur"] - exp["cost_eur"]
    exp["net_per_1000_customers"] = exp["net_value_eur"] / n * 1000
    exp["precision"] = base_all["precision"]
    exp["recall"] = k / n
    exp["roi"] = exp["net_value_eur"] / exp["cost_eur"]
    rows.append({"strategy": f"Random {k} customers (expected)", **exp})
    rule = rule_mask(df)
    if rule.sum() > k:  # respect the same capacity: take the highest-risk rule hits
        keep = np.where(rule)[0][np.argsort(-p[rule])[:k]]
        rule = np.zeros(n, bool)
        rule[keep] = True
    rows.append(_evaluate(df, rule, a, "Portfolio rule (fee increase + 2 complaints)"))
    rows.append(_evaluate(df, _top_k(p, k), a, f"Model: top {k} by churn risk"))
    ev = p * a.save_rate * v - a.offer_cost_eur  # expected net value at decision time
    rows.append(_evaluate(df, _top_k(ev, k), a, f"Model: top {k} by expected net value"))
    rows.append(_evaluate(df, value_gated_mask(df, a), a, "Model: value-gated (only if expected net > 0)"))
    return pd.DataFrame(rows)


def capacity_curve(df, a, shares=None):
    """Net value per 1,000 customers as the retention team contacts more of the base, ranked two ways."""
    shares = shares if shares is not None else np.arange(0.02, 1.0001, 0.02)
    v = customer_value(df, a)
    p = df["p_churn"].values
    ev = p * a.save_rate * v - a.offer_cost_eur
    out = []
    for s in shares:
        k = max(int(round(s * len(df))), 1)
        r = _evaluate(df, _top_k(p, k), a, "risk")
        e = _evaluate(df, _top_k(ev, k), a, "value")
        out.append({"share": s, "net_risk_rank": r["net_per_1000_customers"], "net_value_rank": e["net_per_1000_customers"],
                    "recall_risk_rank": r["recall"]})
    return pd.DataFrame(out)


def gains_curve(df):
    order = np.argsort(-df["p_churn"].values)
    y = df["churned"].values[order]
    return pd.DataFrame({"share_contacted": np.arange(1, len(y) + 1) / len(y), "share_churners_captured": np.cumsum(y) / y.sum()})


def bootstrap_difference(df, a, n_boot=1000, seed=SEED):
    """95% CI for (value-gated model policy) minus (random targeting) in net EUR per 1,000 customers."""
    rng = np.random.default_rng(seed)
    n = len(df)
    diffs, nets = [], []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        k = int(round(a.team_capacity_share * n))
        m = _evaluate(s, value_gated_mask(s, a), a, "m")["net_per_1000_customers"]
        r = _evaluate(s, np.ones(n, bool), a, "all")
        rand = (r["value_protected_eur"] - r["cost_eur"]) * (k / n) / n * 1000
        diffs.append(m - rand)
        nets.append(m)
    return {"diff_mean": float(np.mean(diffs)), "diff_ci_low": float(np.percentile(diffs, 2.5)),
            "diff_ci_high": float(np.percentile(diffs, 97.5)), "net_ci_low": float(np.percentile(nets, 2.5)),
            "net_ci_high": float(np.percentile(nets, 97.5))}


def sensitivity_grid(df, a, save_rates=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40), costs=(15, 30, 45, 60, 90, 120)):
    """Net EUR per 1,000 customers of the value-gated model policy across the two least certain assumptions."""
    from dataclasses import replace
    g = pd.DataFrame(index=list(costs), columns=list(save_rates), dtype=float)
    for c in costs:
        for s in save_rates:
            aa = replace(a, save_rate=s, offer_cost_eur=float(c))
            g.loc[c, s] = _evaluate(df, value_gated_mask(df, aa), aa, "m")["net_per_1000_customers"]
    g.index.name, g.columns.name = "offer_cost_eur", "save_rate"
    return g


def break_even_save_rate(df, a):
    """Smallest save rate at which a FIXED-capacity programme (always contact the top-k by expected value,
    whether or not it pays) breaks even. Below it, forcing the team to fill its quota loses money."""
    from dataclasses import replace
    for s in np.arange(0.01, 1.0, 0.005):
        aa = replace(a, save_rate=float(s))
        v = customer_value(df, aa)
        ev = df["p_churn"].values * aa.save_rate * v - aa.offer_cost_eur
        k = int(round(aa.team_capacity_share * len(df)))
        if _evaluate(df, _top_k(ev, k), aa, "m")["net_value_eur"] >= 0:
            return float(s)
    return float("nan")


def rule_quality(df):
    m = rule_mask(df)
    y = df["churned"].values
    base = y.mean()
    return {"flagged": int(m.sum()), "flagged_share": float(m.mean()), "churn_rate_flagged": float(y[m].mean()),
            "base_churn_rate": float(base), "lift": float(y[m].mean() / base), "recall": float(y[m].sum() / y.sum())}
