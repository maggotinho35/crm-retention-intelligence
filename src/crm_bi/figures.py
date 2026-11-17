"""All README / management figures. Every function takes the results dict `R` built by scripts/run_pipeline.py,
so no number is typed in by hand: change the data or the assumptions and the pictures update.
"""
import matplotlib.ticker as mt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import ASSUMPTIONS as A
from .config import IMG_DIR, PRODUCTS, SEGMENTS
from .viz import BLUE, GOLD, GREEN, GREY, GRID, INK, MUTED, NAVY, RED, SKY, finish, plt

NICE = {"complaint_count_12m": "Complaints (12m)", "monthly_logins": "Monthly logins", "tenure_years": "Tenure (years)",
        "had_fee_increase": "Recent fee increase", "monthly_transactions": "Monthly transactions",
        "segment_young_digital": "Segment: young digital", "segment_retiree": "Segment: retiree",
        "segment_family": "Segment: family", "segment_mass_affluent": "Segment: mass affluent",
        "avg_balance": "Average balance", "num_products": "Number of products", "age": "Age"}
EUR = lambda v, _=None: f"€{v:,.0f}"  # noqa: E731


def _grid(ax, axis="y"):
    getattr(ax, f"{axis}axis").grid(True, color=GRID)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- hero
def hero(R):
    m = R["metrics"]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 14, 4.2, boxstyle="round,pad=0,rounding_size=0.25", fc="#f4f7fa", ec=GRID))
    ax.text(0.5, 3.6, "CRM Retention Intelligence", fontsize=25, fontweight="bold", color=NAVY, va="center")
    ax.text(0.5, 3.02, "From reactive CRM to proactive retention at St. Maggotinho's Bank (fictional)  |  churn scoring  |  "
            "complaint root causes  |  next-best-offer  |  BI architecture", fontsize=9.6, color=MUTED, va="center")
    tiles = [(f"{m['churn_holdout_auc']:.2f}", f"hold-out churn AUC\n95% CI {m['churn_auc_ci'][0]:.2f} to {m['churn_auc_ci'][1]:.2f}", NAVY),
             (f"{m['gated_precision']:.0%}", f"of contacted customers would\nhave left (value-gated, {m['gated_share']:.0%} of base)", GREEN),
             (f"{m['supervised_macro_f1']:.2f}", "complaint-theme macro-F1 once\nagents tag about 150 complaints", GOLD),
             (f"{m['tool_b_win_prob']:.0%}", "of 20,000 weight scenarios still\nfavour the Azure-based architecture", RED)]
    tw, gap, x0 = 3.1, 0.2, 0.5
    for i, (big, small, col) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        ax.add_patch(FancyBboxPatch((x, 0.35), tw, 2.0, boxstyle="round,pad=0,rounding_size=0.15", fc="white", ec=col, lw=2))
        ax.text(x + tw / 2, 1.68, big, ha="center", va="center", fontsize=27, fontweight="bold", color=col)
        ax.text(x + tw / 2, 0.88, small, ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.4)
    finish(fig, IMG_DIR / "hero_banner.png")


# ---------------------------------------------------------------- architecture
def architecture(R):
    fig, ax = plt.subplots(figsize=(13, 7.7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    layers = [("Business users", 0.2, "#fbeadf", RED), ("Delivery", 1.75, "#efe9f8", "#7a5bb5"),
              ("Analytics & ML", 3.3, "#fdf3e1", GOLD), ("Data platform", 4.85, "#e6f1fa", BLUE),
              ("Source systems", 6.4, "#f1f4f7", GREY)]
    for name, y, fc, ec in layers:
        ax.add_patch(FancyBboxPatch((0.15, y), 12.7, 1.35, boxstyle="round,pad=0,rounding_size=0.1", fc=fc, ec=ec, lw=1.2))
        ax.text(0.3, y + 1.2, name, fontsize=9.5, fontweight="bold", color=INK, va="top")

    def box(x, y, w, h, title, body, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.5, body, ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.2))

    # source systems
    for i, (t, b) in enumerate([("Core banking", "transactions,\nbalances"), ("CRM / contact centre", "tickets, complaints,\ninteractions"),
                                ("Digital channels", "app and online\nbanking activity")]):
        box(2.4 + i * 3.5, 6.5, 3.0, 1.0, t, b, GREY)
    # data platform
    box(2.9, 4.95, 3.6, 1.0, "Azure Data Factory", "nightly ETL, data-quality rules", BLUE)
    box(7.0, 4.95, 3.6, 1.0, "Azure Data Lake / SQL DB", "single source of truth, fixed schema", BLUE)
    arrow(6.5, 5.45, 7.0, 5.45)
    # analytics
    for i, (t, b) in enumerate([("Q1  Churn scoring", "logistic regression + SHAP\nretrained monthly"),
                                ("Q2  Complaint themes", "TF-IDF + classifier\nrefreshed weekly"),
                                ("Q3  Next-best-offer", "propensity model + rules\nrefreshed monthly")]):
        box(2.4 + i * 3.5, 3.4, 3.0, 1.1, t, b, GOLD)
    # delivery
    box(2.9, 1.85, 3.6, 1.0, "Power BI", "risk tiers, theme trends,\ncross-sell performance", "#7a5bb5")
    box(7.0, 1.85, 3.6, 1.0, "CRM connector", "scores and offers shown\ninside the agent's screen", "#7a5bb5")
    # users
    for i, (t, b) in enumerate([("Relationship managers", "call value-gated\nhigh-risk customers"),
                                ("Contact-centre agents", "see top-2 offer at\nthe point of contact"),
                                ("CRM management", "fix billing and\nservice processes")]):
        box(2.4 + i * 3.5, 0.3, 3.0, 1.0, t, b, RED)
    for x in (3.9, 7.4, 10.9):
        arrow(x, 6.5, x, 6.0) if x < 7 else None
    arrow(4.7, 6.5, 4.7, 5.95)
    arrow(8.8, 6.5, 8.8, 5.95)
    arrow(8.8, 4.95, 8.8, 4.5)
    arrow(4.2, 4.95, 4.2, 4.5)
    arrow(4.7, 3.4, 4.7, 2.85)
    arrow(8.8, 3.4, 8.8, 2.85)
    arrow(4.7, 1.85, 3.9, 1.3)
    arrow(8.8, 1.85, 7.9, 1.3)
    arrow(8.8, 1.85, 11.4, 1.3)
    ax.text(0.15, 8.45, "Recommended solution architecture (Scenario B: Microsoft stack)", fontsize=14, fontweight="bold",
            color=INK, va="top")
    finish(fig, IMG_DIR / "01_solution_architecture.png",
           "Redrawn from the portfolio's Phase 3 design. Modelling logic stays in portable Python (scikit-learn, "
           "mlxtend, LightGBM) to limit single-vendor lock-in.")


# ---------------------------------------------------------------- data overview
def data_overview(R):
    c, tx = R["customers"], R["tx"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    rate = c.groupby("segment")["churned"].mean().reindex(SEGMENTS) * 100
    bars = a.bar([s.replace("_", "\n") for s in SEGMENTS], rate.values, color=[SKY, BLUE, NAVY, RED], width=0.6)
    a.axhline(c["churned"].mean() * 100, color=MUTED, ls="--", lw=1)
    a.text(-0.45, c["churned"].mean() * 100 + 1, f"overall {c['churned'].mean():.1%}", ha="left", fontsize=9, color=MUTED)
    for r, v in zip(bars, rate.values):
        a.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold",
               bbox=dict(fc="white", ec="none", pad=1))
    a.set_ylabel("Churn rate (%)")
    a.set_title("Retirees churn the most")
    _grid(a)
    hold = tx[PRODUCTS].mean().sort_values() * 100
    b.barh([p.replace("_", " ") for p in hold.index], hold.values, color=BLUE, height=0.65)
    for y, v in enumerate(hold.values):
        b.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=9)
    b.set_xlim(0, 112)
    b.set_xlabel("Customers holding the product (%)")
    b.set_title("Product holding across 1,200 customers")
    _grid(b, "x")
    fig.tight_layout(w_pad=3)
    finish(fig, IMG_DIR / "02_data_overview.png",
           "Synthetic customers generated with seed 42 (see docs/data_dictionary.md). No real bank data is used.")


# ---------------------------------------------------------------- churn models
def churn_models(R):
    t = R["churn"]["table"].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(t))
    ax.errorbar(t["cv_auc_mean"], y + 0.15, xerr=t["cv_auc_sd"], fmt="s", color=GREY, ecolor=GREY, capsize=3,
                label="5-fold CV on training data (mean ± SD)", ms=6)
    ax.errorbar(t["holdout_auc"], y - 0.15, xerr=[t["holdout_auc"] - t["holdout_auc_ci_low"], t["holdout_auc_ci_high"] - t["holdout_auc"]],
                fmt="o", color=NAVY, ecolor=NAVY, capsize=3, label="Untouched hold-out (95% bootstrap CI)", ms=7)
    for yy, v in zip(y, t["holdout_auc"]):
        ax.text(0.945, yy - 0.15, f"{v:.3f}", va="center", fontsize=9, color=NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(t["model"])
    ax.set_xlim(0.70, 0.96)
    ax.set_xlabel("ROC-AUC")
    best = R["churn"]["best_name"]
    ax.set_title("Four churn models are statistically indistinguishable; the simplest one is selected")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, bbox_to_anchor=(0, -0.32), ncol=2)
    _grid(ax, "x")
    fig.tight_layout()
    finish(fig, IMG_DIR / "03_churn_model_comparison.png",
           f"Selected on cross-validation only: {best}. Hold-out n=300 (about 74 churners), so confidence intervals are wide. "
           "Synthetic labels come from a logistic process, which favours linear models.")


def shap_drivers(R):
    s = R["churn"]["shap_imp"].head(8)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cols = [NAVY if v >= s.iloc[-3] else SKY for v in s.values]
    ax.barh([NICE.get(i, i) for i in s.index], s.values, color=cols, height=0.65)
    for yy, v in enumerate(s.values):
        ax.text(v + 0.015, yy, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel(f"Mean |SHAP value| on hold-out customers ({R['churn']['best_name']}, log-odds)")
    ax.set_title("Complaints, low engagement and short tenure drive churn risk")
    _grid(ax, "x")
    fig.tight_layout()
    finish(fig, IMG_DIR / "04_churn_drivers_shap.png",
           "Attribution of the model's predictions, not proof of causation. Segment effects are small once behaviour is known.")


def gains_calibration(R):
    g = R["gains"]
    h = R["churn"]["holdout"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(g["share_contacted"] * 100, g["share_churners_captured"] * 100, color=NAVY, lw=2.2, label="Churn model")
    a.plot([0, 100], [0, 100], color=GREY, ls="--", label="Random targeting")
    for s in (0.1, 0.2):
        v = float(np.interp(s, g["share_contacted"], g["share_churners_captured"])) * 100
        a.scatter([s * 100], [v], color=RED, zorder=3)
        a.annotate(f"top {s:.0%} of customers\ncatch {v:.0f}% of churners", (s * 100, v), xytext=(s * 100 + 8, v - 14 if s == 0.1 else v - 22),
                   fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    a.set_xlabel("Customers contacted, ranked by risk (%)")
    a.set_ylabel("Churners reached (%)")
    a.set_title("Targeting concentrates churners")
    a.legend(frameon=False, loc="lower right")
    _grid(a)
    q = np.quantile(h["p_churn"], np.linspace(0, 1, 6))
    bins = np.clip(np.digitize(h["p_churn"], q[1:-1]), 0, 4)
    pred = [h.loc[bins == i, "p_churn"].mean() for i in range(5)]
    obs = [h.loc[bins == i, "churned"].mean() for i in range(5)]
    b.plot([0, 1], [0, 1], color=GREY, ls="--", label="Perfect calibration")
    b.plot(pred, obs, "o-", color=NAVY, lw=2, label="Hold-out quintiles")
    b.set_xlabel("Predicted churn probability")
    b.set_ylabel("Observed churn rate")
    b.set_title("Scores can be read as probabilities")
    b.legend(frameon=False)
    _grid(b)
    fig.tight_layout(w_pad=3)
    finish(fig, IMG_DIR / "05_gains_and_calibration.png",
           f"Hold-out set, {len(h)} customers. Isotonic calibration applied: {R['churn']['calibration']['calibration_applied']} "
           "(chosen on out-of-fold Brier score of the training split).")


# ---------------------------------------------------------------- business value
def value_by_strategy(R):
    s = R["strat"]
    s = s[~s["strategy"].str.startswith("No action")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, 5))
    y = np.arange(len(s))[::-1]
    vals = s["net_per_1000_customers"].values
    cols = [RED if v < 0 else (GREEN if "gated" in n else BLUE) for v, n in zip(vals, s["strategy"])]
    ax.barh(y, vals, color=cols, height=0.62)
    ax.axvline(0, color=INK, lw=0.8)
    bt = R["boot"]
    gi = int(s.index[s["strategy"].str.contains("gated")][0])
    ax.errorbar(vals[gi], y[gi], xerr=[[vals[gi] - bt["net_ci_low"]], [bt["net_ci_high"] - vals[gi]]], color=INK, capsize=4, lw=1.3)
    span = max(abs(vals).max(), 1)
    xr = span * 0.12
    for yy, v, (_, r) in zip(y, vals, s.iterrows()):
        txt = f"{EUR(v)}\n{int(r['contacted'])} contacted, {r['precision']:.0%} precision" if r["contacted"] else ""
        ax.text(xr, yy, txt, va="center", ha="left", fontsize=8.8, linespacing=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels(s["strategy"], fontsize=9)
    ax.xaxis.set_major_formatter(mt.FuncFormatter(EUR))
    ax.set_xlim(-span * 1.15, span * 0.85)
    ax.set_xlabel("Net value per 1,000 customers (EUR, scenario)")
    ax.set_title("Untargeted outreach destroys value; value-gated targeting makes retention pay")
    _grid(ax, "x")
    fig.tight_layout()
    finish(fig, IMG_DIR / "06_value_by_strategy.png",
           f"ASSUMPTIONS (not measured): offer cost €{A.offer_cost_eur:.0f}, save rate {A.save_rate:.0%}, "
           f"value = (€{A.base_annual_revenue_eur:.0f} + {A.revenue_rate_on_balance:.1%} of balance) x {A.retention_horizon_years:.0f} years, "
           f"capacity {A.team_capacity_share:.0%} of customers. Churners reached are measured on the hold-out. Whisker: 95% bootstrap CI.")


def capacity_sensitivity(R):
    cap, grid = R["cap"], R["grid"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
    a.plot(cap["share"] * 100, cap["net_risk_rank"], color=GREY, lw=2, label="Rank by churn risk")
    a.plot(cap["share"] * 100, cap["net_value_rank"], color=NAVY, lw=2.4, label="Rank by expected net value")
    a.axhline(0, color=INK, lw=0.8)
    j = cap["net_value_rank"].idxmax()
    a.scatter([cap.loc[j, "share"] * 100], [cap.loc[j, "net_value_rank"]], color=GREEN, zorder=3, s=60)
    a.annotate(f"best: contact {cap.loc[j, 'share']:.0%}\n{EUR(cap.loc[j, 'net_value_rank'])} per 1,000", (cap.loc[j, "share"] * 100, cap.loc[j, "net_value_rank"]),
               xytext=(cap.loc[j, "share"] * 100 + 14, cap.loc[j, "net_value_rank"] + 3500), fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    a.set_ylim(top=cap["net_value_rank"].max() + 6500)
    a.yaxis.set_major_formatter(mt.FuncFormatter(EUR))
    a.set_xlabel("Customers contacted (%)")
    a.set_ylabel("Net value per 1,000 customers")
    a.set_title("There is a sweet spot; past it each call loses money")
    a.legend(frameon=False, loc="lower left")
    _grid(a)
    im = b.imshow(grid.values, cmap="Blues", aspect="auto")
    b.set_xticks(range(grid.shape[1]))
    b.set_xticklabels([f"{c:.0%}" for c in grid.columns])
    b.set_yticks(range(grid.shape[0]))
    b.set_yticklabels([f"€{int(i)}" for i in grid.index])
    for i in range(grid.shape[0]):
        for j2 in range(grid.shape[1]):
            v = grid.values[i, j2]
            b.text(j2, i, f"{v / 1000:.1f}k" if abs(v) >= 100 else "0", ha="center", va="center", fontsize=8.5,
                   color="white" if v > grid.values.max() * 0.55 else INK)
    ci, cj = list(grid.index).index(A.offer_cost_eur), [round(c, 2) for c in grid.columns].index(A.save_rate)
    b.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, ec=RED, lw=2.5))
    b.set_xlabel("Share of contacted would-be churners who are saved")
    b.set_ylabel("Cost per retention contact")
    b.set_title("What the result depends on (red box: baseline)")
    for sp in b.spines.values():
        sp.set_visible(False)
    fig.tight_layout(w_pad=3)
    finish(fig, IMG_DIR / "07_capacity_and_sensitivity.png",
           "Net EUR per 1,000 customers for the value-gated policy. Save rate and contact cost are planning assumptions: "
           "validate with a controlled pilot (hold out a random control group) before committing budget.")


# ---------------------------------------------------------------- complaints
def complaints_story(R):
    comp, budget = R["comp"], R["budget"]
    cmp = comp["compare"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    names = [m.replace(" (portfolio baseline)", "\n(portfolio baseline)").replace(" (5-fold CV)", "\n(5-fold CV)") for m in cmp["method"]]
    cols = [GREY, GREY, GREY, GREEN]
    bars = a.barh(names[::-1], cmp["ari_vs_true_category"].values[::-1], color=cols[::-1], height=0.6)
    for r, v in zip(bars, cmp["ari_vs_true_category"].values[::-1]):
        a.text(v + 0.015, r.get_y() + r.get_height() / 2, f"{v:.2f}", va="center", fontweight="bold")
    a.set_xlim(0, 1.05)
    a.set_xlabel("Agreement with true complaint theme (adjusted Rand index)")
    a.set_title("Unsupervised clusters are only partly right")
    _grid(a, "x")
    b.plot(budget["tagged_complaints"], budget["macro_f1_mean"], "o-", color=NAVY, lw=2.2)
    b.fill_between(budget["tagged_complaints"], budget["macro_f1_mean"] - budget["macro_f1_sd"],
                   budget["macro_f1_mean"] + budget["macro_f1_sd"], color=SKY, alpha=0.4)
    for _, r in budget.iterrows():
        b.text(r["tagged_complaints"], r["macro_f1_mean"] - 0.06, f"{r['macro_f1_mean']:.2f}", ha="center", fontsize=8.5)
    b.set_ylim(0.5, 1.0)
    b.set_xlabel("Complaints tagged by agents")
    b.set_ylabel("Macro-F1 on unseen complaints")
    b.set_title("About 150 tagged complaints is enough")
    _grid(b)
    fig.tight_layout(w_pad=3)
    finish(fig, IMG_DIR / "08_complaints_discovery_vs_tagging.png",
           "576 synthetic complaints. Right: mean ± SD over 20 random draws. This supports the portfolio's own recommendation: "
           "better agent tagging, then a supervised model.")


def complaint_money(R):
    t = R["comp"]["theme"].sort_values("share_of_lost_revenue", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(t))
    ax.barh(y + 0.19, t["share_of_complaints"] * 100, height=0.36, color=SKY, label="Share of complaints")
    ax.barh(y - 0.19, t["share_of_lost_revenue"] * 100, height=0.36, color=NAVY, label="Share of revenue lost to churned complainers")
    for yy, (_, r) in zip(y, t.iterrows()):
        ax.text(max(r["share_of_complaints"], r["share_of_lost_revenue"]) * 100 + 0.8, yy, f"churn {r['churn_rate']:.0%}  (n={int(r['complaints'])})", va="center", fontsize=8.8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.set_xlim(0, 48)
    ax.set_xlabel("%")
    ax.set_title("Billing and fee complaints cost more than their volume suggests")
    ax.legend(frameon=False, loc="lower right")
    _grid(ax, "x")
    fig.tight_layout()
    finish(fig, IMG_DIR / "09_complaint_themes_money.png",
           f"Themes from the cross-validated supervised model. Complainers churn at {R['comp']['overall_complainer_churn']:.0%} vs "
           f"{R['customers']['churned'].mean():.0%} overall. The fee-complaint link is a documented generator assumption.")


# ---------------------------------------------------------------- next best offer
def nbo_backtest(R):
    t = R["nbo_bt"].copy()
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    labels = [m.replace(" (portfolio as specified)", "\n(portfolio as specified)").replace(" (ranked by confidence)", "\n(ranked by confidence)") for m in t["method"]]
    cols = [RED if d < 0 else (GREY if abs(d) < 1e-9 else (BLUE if lo <= 0.02 else GREEN)) for d, lo in zip(t["diff_vs_generic"], t["diff_ci_low"])]
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["hit_at_2"] * 100, color=cols, height=0.62)
    for yy, (_, r) in zip(y, t.iterrows()):
        d = "" if abs(r["diff_vs_generic"]) < 1e-9 else f"   {r['diff_vs_generic'] * 100:+.1f} pts vs generic [{r['diff_ci_low'] * 100:+.1f}, {r['diff_ci_high'] * 100:+.1f}]"
        ax.text(r["hit_at_2"] * 100 + 1, yy, f"{r['hit_at_2']:.0%}{d}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 125)
    ax.set_xlabel("hit@2: the hidden product is among the two offers shown (%)")
    ax.set_title("Back-test: the lift-ranked rules lose to the generic script; a propensity model wins")
    _grid(ax, "x")
    fig.tight_layout()
    finish(fig, IMG_DIR / "10_nbo_backtest.png",
           f"Rules mined on 70% of customers, tested on the other 30% ({int(t['n_test_customers'].iloc[0])} customers, one product hidden, "
           "20 repeats). Brackets: 95% paired bootstrap CI. Offline hit rate is not customer acceptance; confirm with an A/B test.")


# ---------------------------------------------------------------- tool selection
def tool_robustness(R):
    from .tool_selection import NAMES, WEIGHTS
    sc, sweep, mc = R["tool_scores"], R["tool_sweep"], R["tool_mc"]
    colors = [GREY, BLUE, RED]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
    bars = a.barh(NAMES[::-1], sc.values[::-1], color=colors[::-1], height=0.55)
    for r, v in zip(bars, sc.values[::-1]):
        a.text(v + 0.05, r.get_y() + r.get_height() / 2, f"{v:.2f}", va="center", fontweight="bold")
    p = mc.loc["weights and scores vary"]
    a.set_xlim(0, 5)
    a.set_xlabel("Weighted score (out of 5)\nWin probability when weights AND scores vary: "
                 f"A {p.iloc[0]:.0%}, B {p.iloc[1]:.0%}, C {p.iloc[2]:.0%}", linespacing=1.6)
    a.set_title("Azure-based stack wins on the chosen weights")
    _grid(a, "x")
    for n, c in zip(NAMES, colors):
        b.plot(sweep.index * 100, sweep[n], color=c, lw=2.2, label=n)
    b.axvline(WEIGHTS[0] * 100, color=INK, ls=":", lw=1.2)
    b.text(WEIGHTS[0] * 100 + 1, 4.9, "chosen weight\n(20%)", fontsize=8.5, va="top")
    flip = R["tool_flip"]
    b.axvline(flip * 100, color=RED, ls="--", lw=1.2)
    b.text(flip * 100 + 1, 4.9, f"A overtakes B\nat {flip:.0%}", fontsize=8.5, color=RED, va="top")
    b.set_xlabel("Weight placed on cost-efficiency (%)")
    b.set_ylabel("Weighted score")
    b.set_ylim(1, 5)
    b.set_title("Decision flips only if cost weighs ~2x more")
    b.legend(frameon=False, fontsize=8.5, loc="lower left")
    _grid(b)
    fig.tight_layout(w_pad=3)
    finish(fig, IMG_DIR / "11_tool_selection_robustness.png",
           "Scores and weights are the portfolio's own judgements (Phase 3). Monte Carlo: 20,000 draws, Dirichlet weights centred on the chosen weights, "
           "score noise SD 0.5. Other weights are rescaled proportionally in the sweep.")


def all_figures(R):
    for f in (hero, architecture, data_overview, churn_models, shap_drivers, gains_calibration, value_by_strategy,
              capacity_sensitivity, complaints_story, complaint_money, nbo_backtest, tool_robustness):
        f(R)
