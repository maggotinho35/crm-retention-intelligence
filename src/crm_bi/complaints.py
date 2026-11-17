"""Q2 / W2: complaint root-cause discovery, and what it takes to make it reliable.

Stage 1 - DISCOVERY (unsupervised, as in the portfolio): TF-IDF + k-means with k=5, plus LSA + k-means and NMF
  as challengers, and a k-sweep. Scored against the hidden true category (ARI) and by silhouette.
Stage 2 - OPERATION (supervised): once agents tag complaints, a TF-IDF + logistic-regression classifier takes
  over. A learning curve answers a practical question: how many complaints must agents tag?
Stage 3 - MONEY: themes (from the cross-validated supervised model) are linked to churn and to the revenue of
  customers who left, so themes are ranked by value and not only by volume.

Honest note: the portfolio's own conclusion ("better agent tagging, then a supervised model") is what the
numbers support. Unsupervised clusters on short, varied text are only moderately aligned with the true themes.
Optional: sentence embeddings (see README, "Extending") need a model download and are not run by default.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import Normalizer

from .business_impact import customer_value
from .config import SEED, BusinessAssumptions

THEME_LABELS = {"billing_fees": "Billing & fees", "app_technical": "App / online banking",
                "wait_time": "Wait times", "card_issues": "Card issues", "communication": "Communication"}


def _vectorize(texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, max_df=0.4, sublinear_tf=True)
    return vec, vec.fit_transform(texts)


def _kmeans(X, k):
    return KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)


def k_sweep(complaints, k_range=range(2, 11)):
    _, X = _vectorize(complaints["complaint_text"])
    rows = []
    for k in k_range:
        km = _kmeans(X, k)
        rows.append({"k": k, "silhouette": silhouette_score(X, km.labels_),
                     "ari": adjusted_rand_score(complaints["true_category"], km.labels_)})
    return pd.DataFrame(rows)


def label_budget_curve(complaints, sizes=(25, 50, 100, 150, 200, 300), repeats=20):
    """Macro-F1 on unseen complaints as a function of how many complaints agents have tagged."""
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, sublinear_tf=True)
    X = vec.fit_transform(complaints["complaint_text"])
    y = complaints["true_category"].values
    rows = []
    for n in sizes:
        f = []
        for s in range(repeats):
            Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n, stratify=y, random_state=s)
            f.append(f1_score(yte, LogisticRegression(max_iter=2000, C=10).fit(Xtr, ytr).predict(Xte), average="macro"))
        rows.append({"tagged_complaints": n, "macro_f1_mean": np.mean(f), "macro_f1_sd": np.std(f)})
    return pd.DataFrame(rows)


def run(complaints: pd.DataFrame, customers: pd.DataFrame, a: BusinessAssumptions, k: int = 5):
    truth = complaints["true_category"]
    vec, X = _vectorize(complaints["complaint_text"])
    km = _kmeans(X, k)
    scores = {"TF-IDF + k-means (portfolio baseline)": (silhouette_score(X, km.labels_), adjusted_rand_score(truth, km.labels_))}
    lsa = Normalizer(copy=False).fit_transform(TruncatedSVD(20, random_state=SEED).fit_transform(X))
    lsa_labels = _kmeans(lsa, k).labels_
    scores["LSA (SVD-20) + k-means"] = (silhouette_score(lsa, lsa_labels), adjusted_rand_score(truth, lsa_labels))
    nmf_labels = NMF(k, init="nndsvda", random_state=SEED, max_iter=500).fit_transform(X).argmax(1)
    scores["NMF topic model"] = (silhouette_score(X, nmf_labels), adjusted_rand_score(truth, nmf_labels))

    # supervised: cross-validated predictions for every complaint (each is predicted by a model that never saw it)
    Xs = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, sublinear_tf=True).fit_transform(
        complaints["complaint_text"])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(LogisticRegression(max_iter=2000, C=10), Xs, truth, cv=cv)
    sup_f1 = f1_score(truth, pred, average="macro")
    scores["Supervised TF-IDF + LogReg (5-fold CV)"] = (np.nan, adjusted_rand_score(truth, pred))
    compare = pd.DataFrame([{"method": m, "silhouette": s, "ari_vs_true_category": r} for m, (s, r) in scores.items()])
    compare["macro_f1"] = [np.nan, np.nan, np.nan, sup_f1]

    # unsupervised cluster table: top terms and purity (one-to-one match to true categories, validation only)
    terms = np.array(vec.get_feature_names_out())
    cats = sorted(truth.unique())
    cont = pd.crosstab(km.labels_, truth).reindex(columns=cats, fill_value=0)
    rows, cols = linear_sum_assignment(-cont.values)
    match = {int(cont.index[r]): THEME_LABELS[cats[c]] for r, c in zip(rows, cols)}
    cluster_table = pd.DataFrame([{"cluster": c, "size": int(cont.loc[c].sum()), "best_matching_true_theme": match.get(c, "-"),
                                   "purity": cont.loc[c].max() / cont.loc[c].sum(),
                                   "top_terms": ", ".join(terms[np.argsort(-km.cluster_centers_[c])[:5]])}
                                  for c in cont.index]).sort_values("size", ascending=False)

    # money view, using the supervised (cross-validated) themes
    df = complaints.copy()
    df["theme"] = pd.Series(pred, index=df.index).map(THEME_LABELS)
    v = pd.Series(customer_value(customers, a), index=customers["customer_id"])
    j = df.merge(customers[["customer_id", "churned"]], on="customer_id")
    j["value_eur"] = j["customer_id"].map(v)
    j["lost_value"] = j["value_eur"] * j["churned"]
    theme = j.groupby("theme").agg(complaints=("customer_id", "size"), churn_rate=("churned", "mean"),
                                   churned=("churned", "sum"), revenue_of_churned_eur=("lost_value", "sum"))
    theme["share_of_complaints"] = theme["complaints"] / theme["complaints"].sum()
    theme["share_of_lost_revenue"] = theme["revenue_of_churned_eur"] / theme["revenue_of_churned_eur"].sum()
    theme = theme.sort_values("complaints", ascending=False)
    theme["cum_share"] = theme["share_of_complaints"].cumsum()
    return dict(clusters=df, theme=theme, compare=compare, cluster_table=cluster_table,
                silhouette=scores["TF-IDF + k-means (portfolio baseline)"][0],
                ari=scores["TF-IDF + k-means (portfolio baseline)"][1], supervised_macro_f1=float(sup_f1),
                overall_complainer_churn=float(j["churned"].mean()))
