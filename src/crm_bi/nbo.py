"""Q3 / W3: next-best-offer.

Portfolio method: Apriori association rules (min support 8%, min lift 1.1). Here the rules are also
BACK-TESTED, so "beats a generic script" is a measured claim instead of an assertion:

  1. Split customers 70/30. Mine rules on the 70% only.
  2. For each test customer holding >= 2 non-checking products, hide one product at random.
  3. Recommend the top-2 products from (a) the rule engine exactly as specified (ranked by lift), (a2) the same
     rules with a popularity back-off and confidence ranking, (b) a logistic-regression propensity model,
     (c) the generic script (the two most popular products the customer does not hold), (d) random.
  4. hit@2 = share of test customers whose hidden product is among the two recommended.
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import PRODUCTS, SEED

CORE = "checking_account"  # held by ~97% of customers: carries no recommendation signal


def mine_rules(baskets: pd.DataFrame, min_support=0.08, min_lift=1.1):
    freq = apriori(baskets.astype(bool), min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].copy()
    rules["antecedent"] = rules["antecedents"].apply(lambda s: next(iter(s)))
    rules["consequent"] = rules["consequents"].apply(lambda s: next(iter(s)))
    keep = ["antecedent", "consequent", "support", "confidence", "lift"]
    return rules[keep].sort_values("lift", ascending=False).reset_index(drop=True)


def _rule_scores(held: np.ndarray, rules: pd.DataFrame, key: str):
    """Best matching rule per consequent. key='lift' (portfolio as specified) or 'confidence'."""
    s = np.zeros(len(PRODUCTS))
    for _, r in rules.iterrows():
        if held[PRODUCTS.index(r["antecedent"])]:
            j = PRODUCTS.index(r["consequent"])
            s[j] = max(s[j], r[key])
    return s


def backtest(tx: pd.DataFrame, customers: pd.DataFrame, n_repeats=20, seed=SEED):
    rng = np.random.default_rng(seed)
    base = tx[PRODUCTS]
    train_ids, test_ids = train_test_split(np.arange(len(tx)), test_size=0.30, random_state=seed)
    rules = mine_rules(base.iloc[train_ids])
    pop = base.iloc[train_ids].mean().values
    seg = pd.get_dummies(tx["segment"], dtype=int).values
    age = ((customers["age"].values - customers["age"].mean()) / customers["age"].std())[:, None]
    ctx = np.hstack([seg, age])
    H = base.values
    movable = [PRODUCTS.index(p) for p in PRODUCTS if p != CORE]

    # propensity models trained on training customers only (inputs = other holdings + segment + age)
    models = {}
    for j in movable:
        Xj = np.hstack([np.delete(H, j, axis=1), ctx])
        models[j] = LogisticRegression(max_iter=1000, C=0.5).fit(Xj[train_ids], H[train_ids, j])

    res = {"Association rules, ranked by lift (portfolio as specified)": [],
           "Rules + popularity back-off (ranked by confidence)": [], "Propensity model (LogReg)": [],
           "Generic script (most popular)": [], "Random": []}
    eligible = [i for i in test_ids if H[i, movable].sum() >= 2]
    per_cust = {m: np.zeros(len(eligible)) for m in res}   # mean hit over repeats, per customer (for paired CIs)
    for _ in range(n_repeats):
        hits = {m: 0 for m in res}
        for ci, i in enumerate(eligible):
            held = H[i].copy()
            hide = rng.choice([j for j in movable if held[j] == 1])
            held[hide] = 0
            cand = [j for j in movable if held[j] == 0]
            # (a) rule engine
            sc = _rule_scores(held, rules, "lift")
            top_rules = sorted(cand, key=lambda j: (-sc[j], -pop[j]))[:2]
            # (a2) hybrid: confidence of the matching rule if there is one, otherwise the product's base rate
            cf = _rule_scores(held, rules, "confidence")
            top_hybrid = sorted(cand, key=lambda j: -(cf[j] if cf[j] > 0 else pop[j]))[:2]
            # (b) propensity model
            ph = {j: models[j].predict_proba(np.hstack([np.delete(held, j), ctx[i]])[None, :])[0, 1] for j in cand}
            top_prop = sorted(cand, key=lambda j: -ph[j])[:2]
            # (c) generic script and (d) random
            top_pop = sorted(cand, key=lambda j: -pop[j])[:2]
            top_rand = list(rng.choice(cand, size=min(2, len(cand)), replace=False))
            for m, top in zip(res, [top_rules, top_hybrid, top_prop, top_pop, top_rand]):
                hits[m] += int(hide in top)
                per_cust[m][ci] += int(hide in top) / n_repeats
        for m in res:
            res[m].append(hits[m] / len(eligible))
    out = pd.DataFrame({"method": list(res), "hit_at_2": [np.mean(v) for v in res.values()],
                        "hit_at_2_sd_over_repeats": [np.std(v) for v in res.values()]})
    out["n_test_customers"] = len(eligible)
    # binomial standard error for the headline figure (sampling noise from the finite test set)
    out["binomial_se"] = np.sqrt(out["hit_at_2"] * (1 - out["hit_at_2"]) / len(eligible))
    out["relative_to_generic"] = out["hit_at_2"] / out.loc[out.method.str.startswith("Generic"), "hit_at_2"].iloc[0]
    # paired bootstrap over customers: difference in hit@2 versus the generic script
    gen = per_cust[[m for m in res if m.startswith("Generic")][0]]
    lo, hi, d = [], [], []
    brng = np.random.default_rng(seed)
    idx = brng.integers(0, len(eligible), (2000, len(eligible)))
    for m in res:
        diff = (per_cust[m][idx] - gen[idx]).mean(1)
        d.append(per_cust[m].mean() - gen.mean())
        lo.append(np.percentile(diff, 2.5))
        hi.append(np.percentile(diff, 97.5))
    out["diff_vs_generic"], out["diff_ci_low"], out["diff_ci_high"] = d, lo, hi
    return rules, out


def cross_sell_value(backtest_table: pd.DataFrame, a, contacts: int = 1000) -> pd.DataFrame:
    """Scenario translation of the back-test into euros per `contacts` service contacts.

    ASSUMPTION: the acceptance rate scales with the offline hit rate relative to the generic script
    (config.cross_sell_acceptance_baseline x hit-rate ratio). Offline hit rate is not the same thing as
    customer acceptance, so this is an upper-level planning scenario that only a live A/B test can confirm.
    """
    gen = backtest_table.loc[backtest_table.method.str.startswith("Generic"), "hit_at_2"].iloc[0]
    out = backtest_table[["method", "hit_at_2", "diff_vs_generic", "diff_ci_low", "diff_ci_high"]].copy()
    out["relative_to_generic"] = out["hit_at_2"] / gen
    base_accepts = contacts * a.cross_sell_acceptance_baseline
    out["accepted_offers"] = base_accepts * out["relative_to_generic"]
    out["extra_offers_vs_generic"] = out["accepted_offers"] - base_accepts
    out["extra_margin_eur"] = out["extra_offers_vs_generic"] * a.cross_sell_margin_eur
    out["extra_margin_ci_low_eur"] = base_accepts * (out["diff_ci_low"] / gen) * a.cross_sell_margin_eur
    out["extra_margin_ci_high_eur"] = base_accepts * (out["diff_ci_high"] / gen) * a.cross_sell_margin_eur
    return out
