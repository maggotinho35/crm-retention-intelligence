"""Synthetic data for St. Maggotinho's Bank (fictional). Documented, seeded and reproducible.

Design follows the portfolio's dataset appendix: four behavioural segments, a churn label drawn from
a logistic model of engagement, tenure, complaints, fee increase and product count (so it is learnable
but noisy), five complaint themes with varied phrasing, and segment-specific product holding.

This generator is a re-implementation calibrated to the statistics reported in the appendix. It is NOT
the original generator, so individual rows and model scores will differ slightly from the PDF.
"""
import numpy as np
import pandas as pd

from .config import COMPLAINT_CATEGORIES, N_CUSTOMERS, PRODUCTS, SEED, SEGMENTS

SEGMENT_PROPS = [0.303, 0.292, 0.204, 0.201]

# mean/sd of the continuous attributes per segment: (age, tenure, logins, transactions, balance_mu, balance_sigma)
SEG_PARAMS = {
    "young_digital": dict(age=(27, 5), tenure=(3.0, 2.0), logins=(22, 5), tx=(22, 5), bal=(7.4, 0.75)),
    "family": dict(age=(42, 7), tenure=(6.5, 3.5), logins=(14, 5), tx=(19, 5), bal=(7.9, 0.70)),
    "mass_affluent": dict(age=(46, 9), tenure=(8.5, 4.5), logins=(12, 5), tx=(18, 6), bal=(8.8, 0.55)),
    "retiree": dict(age=(66, 7), tenure=(10.5, 5.5), logins=(5.5, 3.5), tx=(10, 4.5), bal=(8.4, 0.65)),
}

# probability of holding each product by segment, in PRODUCTS order
PRODUCT_PROPS = {
    "young_digital": [0.97, 0.60, 0.40, 0.10, 0.03, 0.12, 0.12, 0.70],
    "family": [0.97, 0.72, 0.52, 0.22, 0.34, 0.20, 0.38, 0.35],
    "mass_affluent": [0.97, 0.80, 0.60, 0.15, 0.17, 0.52, 0.42, 0.25],
    "retiree": [0.95, 0.78, 0.38, 0.12, 0.05, 0.32, 0.45, 0.06],
}
# co-holding structure (what makes association rules findable): extra lift when a "trigger" product is held
PRODUCT_COUPLING = [("investment_fund", "insurance_bundle", 0.22), ("mortgage", "credit_card", 0.20),
                    ("personal_loan", "credit_card", 0.16), ("insurance_bundle", "savings_account", 0.10)]

COMPLAINT_TEMPLATES = {
    "billing_fees": [
        "My account was charged twice for the same monthly fee.",
        "I was hit with an unexpected maintenance fee that nobody explained.",
        "The overdraft charge on my statement is wrong and unclear.",
        "Why did my account fees go up without any notice?",
        "There is a billing error on my last statement and I want a refund.",
        "The bank took a fee I never agreed to.",
        "My statement shows a charge I do not recognise.",
        "The new account fee is much higher than what was promised."],
    "app_technical": [
        "Online banking website is extremely slow and times out.",
        "The mobile app keeps crashing when I try to log in.",
        "I cannot complete a transfer in the app, it shows an error.",
        "Login to online banking fails after the latest update.",
        "The app freezes on the payment screen every time.",
        "Two-factor code never arrives so I cannot get into the app.",
        "Online portal is down again and I cannot see my balance.",
        "The app logs me out in the middle of a transfer."],
    "wait_time": [
        "I waited forty minutes on the phone before anyone answered.",
        "The wait at the branch was far too long.",
        "Waiting time at the counter is unacceptable.",
        "Customer service wait times are terrible, it took days to reply.",
        "I was on hold for an hour and the wait ended with a dropped call.",
        "The wait for my loan paperwork to be processed took weeks.",
        "Every visit to the branch means a long wait in the queue.",
        "Staff took too long, I waited ages for a simple transaction."],
    "card_issues": [
        "The card reader kept rejecting my chip card at the store.",
        "My debit card was blocked for no reason while travelling.",
        "My new credit card has still not arrived after three weeks.",
        "Card payment declined even though I have enough funds.",
        "The contactless payment on my card stopped working.",
        "My card was swallowed by the cash machine.",
        "I was refused at the till because the card was flagged.",
        "The replacement card came with the wrong PIN letter."],
    "communication": [
        "Nobody told me what was happening with my application.",
        "I received conflicting information from two different advisers.",
        "No one called me back as promised.",
        "I was never informed about the change to my account terms.",
        "The advisor did not explain the options clearly.",
        "I keep getting passed between departments with no answer.",
        "Nobody explained why my request was rejected.",
        "I only found out about the change by accident."],
}
VAGUE = ["Something is wrong with my account and I need help.", "I am not happy with the service I received.",
         "This has been a poor experience overall.", "I would like someone to look into my account please."]
PREFIX = [""] * 8 + ["Again, ", "Last week "]
SUFFIX = [""] * 7 + [" Please fix this.", " This is unacceptable.", " I expect a reply."]


def _compose_text(cat, rng):
    """Natural variation: optional opener, optional closer, and ~8% deliberately vague complaints."""
    if rng.random() < 0.08:
        core = rng.choice(VAGUE)
    else:
        core = rng.choice(COMPLAINT_TEMPLATES[cat])
    pre = rng.choice(PREFIX)
    if pre and core[0].isupper() and not pre.endswith(". "):
        core = core[0].lower() + core[1:]
    return f"{pre}{core}{rng.choice(SUFFIX)}"


COMPLAINT_PROPS = [0.267, 0.250, 0.209, 0.180, 0.094]


def _z(x):
    return (x - x.mean()) / x.std()


def generate(seed: int = SEED, n: int = N_CUSTOMERS):
    rng = np.random.default_rng(seed)
    seg = rng.choice(SEGMENTS, size=n, p=SEGMENT_PROPS)
    cols = {k: np.zeros(n) for k in ["age", "tenure_years", "monthly_logins", "monthly_transactions", "avg_balance"]}
    for s in SEGMENTS:
        idx = seg == s
        k = idx.sum()
        p = SEG_PARAMS[s]
        cols["age"][idx] = rng.normal(*p["age"], k)
        cols["tenure_years"][idx] = rng.gamma(2.2, p["tenure"][0] / 2.2, k)
        cols["monthly_logins"][idx] = rng.normal(*p["logins"], k)
        cols["monthly_transactions"][idx] = rng.normal(*p["tx"], k)
        cols["avg_balance"][idx] = rng.lognormal(*p["bal"], k)
    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)], "segment": seg,
        "age": np.clip(cols["age"], 18, 90).round().astype(int),
        "tenure_years": np.clip(cols["tenure_years"], 0.2, 35).round(1),
        "monthly_logins": np.clip(cols["monthly_logins"], 0, 31).round(1),
        "monthly_transactions": np.clip(cols["monthly_transactions"], 0, 36).round().astype(int),
        "avg_balance": np.clip(cols["avg_balance"], 200, 60000).round(2)})

    # product holdings (before num_products so the count is consistent)
    hold = np.zeros((n, len(PRODUCTS)), dtype=int)
    for s in SEGMENTS:
        idx = np.where(seg == s)[0]
        hold[idx] = rng.random((len(idx), len(PRODUCTS))) < np.array(PRODUCT_PROPS[s])
    for a, b, bump in PRODUCT_COUPLING:
        ia, ib = PRODUCTS.index(a), PRODUCTS.index(b)
        add = (hold[:, ia] == 1) & (hold[:, ib] == 0) & (rng.random(n) < bump)
        hold[add, ib] = 1
    hold[:, 0] = np.where(hold.sum(1) == 0, 1, hold[:, 0])
    df["num_products"] = hold.sum(1)

    # fee increase and complaints (complaints higher when a fee increase happened)
    fee = (rng.random(n) < 0.22).astype(int)
    lam = 0.56 + 0.55 * fee + 0.10 * (df["monthly_logins"] < 8)
    complaints = np.clip(rng.poisson(lam), 0, 5)
    df["complaint_count_12m"] = complaints
    df["had_fee_increase"] = fee

    # churn: logistic model on standardised drivers (+ noise), so it is learnable but never perfect
    score = (-0.55 * _z(df["monthly_logins"]) - 0.30 * _z(df["monthly_transactions"])
             - 0.48 * _z(df["tenure_years"]) + 0.72 * _z(df["complaint_count_12m"])
             + 0.60 * fee - 0.10 * _z(df["num_products"]) + 0.05 * _z(df["age"]) - 0.04 * _z(df["avg_balance"]))
    seg_shift = pd.Series(seg).map({"young_digital": 0.0, "family": 0.12, "mass_affluent": 0.30, "retiree": 0.0}).values
    score = score + seg_shift + rng.normal(0, 0.35, n)
    u = rng.random(n)
    # intercept chosen by bisection so the realised churn rate matches the documented 24.6%
    lo, hi = -8.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        rate = (u < 1 / (1 + np.exp(-(1.9 * score + mid)))).mean()
        lo, hi = (mid, hi) if rate < 0.246 else (lo, mid)
    df["churned"] = (u < 1 / (1 + np.exp(-(1.9 * score + mid)))).astype(int)

    # complaint records: one free-text complaint per customer who complained
    # a recent fee increase makes a billing/fee complaint more likely (documented generator assumption)
    comp_rows = []
    base = np.array(COMPLAINT_PROPS) * np.array([0.78, 1, 1, 1, 1])
    fee_p = base * np.array([2.6, 1, 1, 1, 1])
    base, fee_p = base / base.sum(), fee_p / fee_p.sum()
    for i in np.where(complaints > 0)[0]:
        cat = rng.choice(COMPLAINT_CATEGORIES, p=fee_p if fee[i] else base)
        comp_rows.append((df.loc[i, "customer_id"], cat, _compose_text(cat, rng)))
    complaints_df = pd.DataFrame(comp_rows, columns=["customer_id", "true_category", "complaint_text"])

    tx = pd.DataFrame(hold, columns=PRODUCTS)
    tx.insert(0, "segment", df["segment"].values)
    tx.insert(0, "customer_id", df["customer_id"].values)
    return df, complaints_df, tx


if __name__ == "__main__":
    c, k, t = generate()
    print(c.churned.mean().round(3), len(k))
    print(c.groupby("segment").churned.mean().round(3).to_dict())
    print(t[PRODUCTS].mean().round(3).to_dict())
