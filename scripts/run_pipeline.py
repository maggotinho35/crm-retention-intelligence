"""End-to-end pipeline: data -> churn -> business value -> complaints -> next-best-offer -> tool choice -> figures.

    python scripts/run_pipeline.py                 # everything (about 1-2 minutes)
    python scripts/run_pipeline.py --no-figures    # tables and metrics only
    python scripts/run_pipeline.py --boot 300      # fewer bootstrap draws for a quick smoke test

Outputs: data/*.csv, outputs/*.csv, outputs/metrics.json, outputs/results_summary.md, images/*.png
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crm_bi import business_impact as bi  # noqa: E402
from crm_bi import churn, complaints, nbo  # noqa: E402
from crm_bi import tool_selection as ts  # noqa: E402
from crm_bi.config import ASSUMPTIONS as A  # noqa: E402
from crm_bi.config import DATA_DIR, IMG_DIR, OUT_DIR, SEED  # noqa: E402
from crm_bi.data_generation import generate  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    d = df.copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws for the business-value CI")
    args = ap.parse_args()
    t0 = time.time()
    for d in (DATA_DIR, OUT_DIR, IMG_DIR):
        d.mkdir(exist_ok=True)

    print("1/6 generating synthetic data")
    customers, complaints_df, tx = generate()
    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    complaints_df.to_csv(DATA_DIR / "complaints.csv", index=False)
    tx.to_csv(DATA_DIR / "transactions.csv", index=False)

    print("2/6 churn models")
    ch = churn.run(customers)
    ch["table"].to_csv(OUT_DIR / "churn_model_comparison.csv", index=False)
    ch["shap_imp"].rename("mean_abs_shap").to_csv(OUT_DIR / "churn_shap_importance.csv")
    ch["holdout"].to_csv(OUT_DIR / "holdout_predictions.csv", index=False)
    hold = ch["holdout"]

    print("3/6 business value")
    strat = bi.compare_strategies(hold, A)
    cap = bi.capacity_curve(hold, A)
    grid = bi.sensitivity_grid(hold, A)
    boot = bi.bootstrap_difference(hold, A, n_boot=args.boot)
    breakeven = bi.break_even_save_rate(hold, A)
    gains = bi.gains_curve(hold)
    strat.to_csv(OUT_DIR / "strategy_comparison.csv", index=False)
    cap.to_csv(OUT_DIR / "capacity_curve.csv", index=False)
    grid.to_csv(OUT_DIR / "sensitivity_grid.csv")
    rule_full, rule_hold = bi.rule_quality(customers), bi.rule_quality(hold)

    print("4/6 complaints")
    comp = complaints.run(complaints_df, customers, A)
    sweep = complaints.k_sweep(complaints_df)
    budget = complaints.label_budget_curve(complaints_df)
    comp["compare"].to_csv(OUT_DIR / "complaint_method_comparison.csv", index=False)
    comp["theme"].to_csv(OUT_DIR / "complaint_themes_value.csv")
    comp["cluster_table"].to_csv(OUT_DIR / "complaint_clusters_unsupervised.csv", index=False)
    sweep.to_csv(OUT_DIR / "complaint_k_sweep.csv", index=False)
    budget.to_csv(OUT_DIR / "complaint_label_budget.csv", index=False)

    print("5/6 next-best-offer")
    rules, bt = nbo.backtest(tx, customers)
    xs = nbo.cross_sell_value(bt, A)
    rules.to_csv(OUT_DIR / "nbo_rules.csv", index=False)
    bt.to_csv(OUT_DIR / "nbo_backtest.csv", index=False)
    xs.to_csv(OUT_DIR / "nbo_cross_sell_value.csv", index=False)

    print("6/6 tool selection")
    tool_scores = ts.weighted_scores()
    tool_mc = ts.monte_carlo()
    tool_sweep, tool_flip = ts.cost_weight_breakeven()
    pd.concat([tool_scores.rename("weighted_score"), tool_mc.T], axis=1).to_csv(OUT_DIR / "tool_selection.csv")

    gated = strat[strat["strategy"].str.contains("gated")].iloc[0]
    rule_row = strat[strat["strategy"].str.startswith("Portfolio rule")].iloc[0]
    random_row = strat[strat["strategy"].str.startswith("Random")].iloc[0]
    best_row = ch["table"].iloc[0]
    metrics = {
        "assumptions": asdict(A), "seed": SEED, "n_customers": len(customers), "churn_rate": float(customers.churned.mean()),
        "n_complaints": len(complaints_df), "n_holdout": len(hold), "n_holdout_churners": int(hold.churned.sum()),
        "churn_selected_model": ch["best_name"], "churn_holdout_auc": float(best_row["holdout_auc"]),
        "churn_auc_ci": [float(best_row["holdout_auc_ci_low"]), float(best_row["holdout_auc_ci_high"])],
        "churn_pr_auc": float(best_row["holdout_pr_auc"]), "churn_cv_auc": float(best_row["cv_auc_mean"]),
        "calibration": ch["calibration"],
        "gated_contacted": int(gated["contacted"]), "gated_share": float(gated["contacted"] / len(hold)),
        "gated_precision": float(gated["precision"]), "gated_recall": float(gated["recall"]),
        "gated_net_per_1000": float(gated["net_per_1000_customers"]), "gated_roi": float(gated["roi"]),
        "random_net_per_1000": float(random_row["net_per_1000_customers"]),
        "blanket_net_per_1000": float(strat[strat["strategy"].str.startswith("Blanket")].iloc[0]["net_per_1000_customers"]),
        "rule_precision": float(rule_row["precision"]), "rule_net_per_1000": float(rule_row["net_per_1000_customers"]),
        "bootstrap_vs_random": boot, "break_even_save_rate_fixed_capacity": breakeven,
        "rule_quality_all_customers": rule_full, "rule_quality_holdout": rule_hold,
        "top20_recall": float(np.interp(0.2, gains["share_contacted"], gains["share_churners_captured"])),
        "top10_recall": float(np.interp(0.1, gains["share_contacted"], gains["share_churners_captured"])),
        "complaint_ari_kmeans": float(comp["ari"]), "complaint_silhouette_kmeans": float(comp["silhouette"]),
        "supervised_macro_f1": comp["supervised_macro_f1"],
        "complaint_f1_at_150": float(budget.loc[budget.tagged_complaints == 150, "macro_f1_mean"].iloc[0]),
        "nbo_hit_at_2": dict(zip(bt["method"], bt["hit_at_2"].round(4))),
        "tool_scores": tool_scores.to_dict(), "tool_b_win_prob": float(tool_mc.loc["weights and scores vary"].iloc[1]),
        "tool_b_win_prob_weights_only": float(tool_mc.loc["weights vary"].iloc[1]), "tool_cost_weight_flip": tool_flip,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    R = dict(customers=customers, complaints=complaints_df, tx=tx, churn=ch, strat=strat, cap=cap, grid=grid, boot=boot,
             gains=gains, comp=comp, sweep=sweep, budget=budget, rules=rules, nbo_bt=bt, xsell=xs, tool_scores=tool_scores,
             tool_mc=tool_mc, tool_sweep=tool_sweep, tool_flip=tool_flip, metrics=metrics)

    summary = ["# Results summary (auto-generated by scripts/run_pipeline.py)\n",
               "All euro figures are scenarios built on the assumptions in `src/crm_bi/config.py`.\n",
               "## Churn models\n", md_table(ch["table"]), "\n## Targeting strategies on the hold-out\n",
               md_table(strat[["strategy", "contacted", "churners_reached", "precision", "recall", "net_per_1000_customers", "roi"]]),
               "\n## Complaint methods\n", md_table(comp["compare"]), "\n## Complaint themes and value\n",
               md_table(comp["theme"].reset_index()), "\n## Next-best-offer back-test\n",
               md_table(bt[["method", "hit_at_2", "diff_vs_generic", "diff_ci_low", "diff_ci_high"]]),
               "\n## Tool selection\n", md_table(pd.concat([tool_scores.rename("weighted_score"), tool_mc.T], axis=1).reset_index())]
    (OUT_DIR / "results_summary.md").write_text("\n".join(summary))

    if not args.no_figures:
        from crm_bi import figures
        figures.all_figures(R)
    print(f"done in {time.time() - t0:.0f}s -> outputs/metrics.json")


if __name__ == "__main__":
    main()
