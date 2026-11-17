"""Q1 / W1: churn-risk scoring.

Beyond the portfolio's single Random Forest, this module:
  * compares Logistic Regression, Random Forest, Gradient Boosting (sklearn) and LightGBM,
  * uses stratified 5-fold CV on the training split and ONE evaluation on the untouched 25% hold-out,
  * puts bootstrap confidence intervals on the hold-out AUC,
  * calibrates probabilities (isotonic) so a score of 0.30 really means ~30%, which the euro maths needs,
  * explains the final model with SHAP.
"""
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def features(customers: pd.DataFrame):
    X = pd.get_dummies(customers.drop(columns=["customer_id", "churned"]), columns=["segment"], dtype=int)
    return X, customers["churned"].values


def model_zoo():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "Random Forest (portfolio baseline)": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=8, min_child_samples=20,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                                   verbose=-1),
    }


def bootstrap_auc_ci(y, p, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run(customers: pd.DataFrame):
    X, y = features(customers)
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(X)), test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted, hold_p = [], {}, {}
    for name, model in model_zoo().items():
        cv_auc = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc")
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        lo, hi = bootstrap_auc_ci(yte, p)
        rows.append({"model": name, "cv_auc_mean": cv_auc.mean(), "cv_auc_sd": cv_auc.std(),
                     "holdout_auc": roc_auc_score(yte, p), "holdout_auc_ci_low": lo, "holdout_auc_ci_high": hi,
                     "holdout_pr_auc": average_precision_score(yte, p), "holdout_brier": brier_score_loss(yte, p)})
        fitted[name], hold_p[name] = model, p
    table = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)

    # model selection on cross-validation only; the hold-out is never used to choose
    best_name = table.loc[0, "model"]
    best = fitted[best_name]
    # Calibration: use isotonic calibration only if it lowers the OUT-OF-FOLD Brier score on the training
    # split. The hold-out is not consulted for this choice.
    oof_raw = cross_val_predict(model_zoo()[best_name], Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    cal_est = CalibratedClassifierCV(model_zoo()[best_name], method="isotonic", cv=3)
    oof_cal = cross_val_predict(cal_est, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    use_cal = brier_score_loss(ytr, oof_cal) < brier_score_loss(ytr, oof_raw)
    cal = CalibratedClassifierCV(model_zoo()[best_name], method="isotonic", cv=5).fit(Xtr, ytr)
    p_cal = cal.predict_proba(Xte)[:, 1]
    p_final = p_cal if use_cal else hold_p[best_name]
    calib = {"model": best_name, "oof_brier_raw": brier_score_loss(ytr, oof_raw),
             "oof_brier_calibrated": brier_score_loss(ytr, oof_cal), "calibration_applied": bool(use_cal),
             "holdout_brier_final": brier_score_loss(yte, p_final), "holdout_auc_final": roc_auc_score(yte, p_final)}

    # SHAP on the selected model (tree or linear)
    explain_X = Xte.copy()
    shap_values = _shap(best, best_name, Xtr, explain_X)
    shap_imp = pd.Series(np.abs(shap_values).mean(0), index=explain_X.columns).sort_values(ascending=False)

    holdout = customers.iloc[idx_te].copy()
    holdout["p_churn_raw"] = hold_p[best_name]
    holdout["p_churn"] = p_final
    for name, p in hold_p.items():
        holdout["p_" + name.split(" (")[0].lower().replace(" ", "_")] = p
    return dict(table=table, best_name=best_name, calibration=calib, holdout=holdout, shap_imp=shap_imp,
                shap_values=shap_values, explain_X=explain_X, y_test=yte)


def _shap(model, name, Xtr, Xexp):
    if name.startswith("Logistic"):
        scaler, lr = model.steps[0][1], model.steps[1][1]
        masker = shap.maskers.Independent(scaler.transform(Xtr), max_samples=len(Xtr))
        sv = shap.LinearExplainer(lr, masker).shap_values(scaler.transform(Xexp))
        return np.asarray(sv)
    if name.startswith("Gradient"):
        f = lambda d: model.predict_proba(d)[:, 1]  # noqa: E731
        return shap.Explainer(f, Xtr.sample(100, random_state=SEED))(Xexp).values
    sv = shap.TreeExplainer(model).shap_values(Xexp)
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv
