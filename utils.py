"""
Utility helpers for clinical binary-classification evaluation
=============================================================

Model-agnostic functions for AUC reporting, threshold-policy
selection, calibration helpers, and validation visualisations.

Adapted from the IE-ML-for-Healthcare RAI reference repo with
**generic clinical labels** so the same file works for any
binary healthcare risk-stratification task.

Functions
---------
- positive_scores         – P(positive) or decision-function scores
- auc_report              – ROC AUC, PR AUC, prevalence, lift + plots
- tradeoff_table          – precision / recall / confusion at many thresholds
- pick_threshold_cost     – minimise illustrative FP/FN harm weights
- pick_threshold_recall_floor – max precision ≥ recall floor
- pick_threshold_workload – max TP/1 000 under alert budget
- summary_at_threshold    – one-row summary at a single threshold
- wilson_interval         – Wilson score CI for a binomial proportion
- plot_recall_floor_curves
- plot_cumulative_recall_at_threshold
- plot_topk_at_threshold
- make_thresholded_estimator / ThresholdedEstimator
- init_rai_dependencies   – safe import of optional RAI packages
- subgroup_report         – subgroup recall/precision/alerts at a fixed threshold (+ Wilson CI)
- expected_calibration_error / calibration_summary / plot_reliability – calibration metrics and plot
- bootstrap_metric_ci / threshold_stability – uncertainty of AUC, recall and alert volume
- tiered_alert_table      – two-tier (intensive / light-touch) operating point
- group_recall_floor_thresholds / apply_group_thresholds – per-group thresholds
- ppv_under_prevalence    – PPV / workload if event prevalence shifts
- leakage_audit           – computed leakage checks
- subgroup_auc, error_tree_paths, whatif_risk_by_arm, cf_summary – RAI-result helpers
- monitoring_check        – one monitoring cycle with alert triggers
- NumericArmAdapter / encode_arm – lets Fairlearn call a model with a string treatment column
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
)
from scipy.stats import norm

__all__ = [
    "positive_scores",
    "auc_report",
    "tradeoff_table",
    "pick_threshold_cost",
    "pick_threshold_recall_floor",
    "pick_threshold_workload",
    "summary_at_threshold",
    "wilson_interval",
    "plot_recall_floor_curves",
    "plot_cumulative_recall_at_threshold",
    "plot_topk_at_threshold",
    "make_thresholded_estimator",
    "init_rai_dependencies",
    "subgroup_report",
    "expected_calibration_error", "calibration_summary", "plot_reliability",
    "bootstrap_metric_ci", "threshold_stability", "tiered_alert_table",
    "group_recall_floor_thresholds", "apply_group_thresholds", "ppv_under_prevalence",
    "leakage_audit", "subgroup_auc", "error_tree_paths", "whatif_risk_by_arm",
    "cf_summary", "monitoring_check", "NumericArmAdapter", "encode_arm",
]


# ──────────────────────────────────────────────
#  Confidence interval
# ──────────────────────────────────────────────

def wilson_interval(successes: int, total: int,
                    confidence: float = 0.95) -> Tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
              / denominator)
    return (centre - margin, centre + margin)


# ──────────────────────────────────────────────
#  Optional RAI imports
# ──────────────────────────────────────────────

def init_rai_dependencies():
    """Attempt to import optional Responsible AI dependencies.

    Returns
    -------
    Tuple[Dict[str, bool], Dict[str, object]]
        (status_flags, imported_objects)
    """
    status = {
        "_RAI": False,
        "_INTERPRET": False,
        "_ERRANALYSIS": False,
        "_FAIRLEARN": False,
    }
    objects = {
        "RAIInsights": None,
        "FeatureMetadata": None,
        "ResponsibleAIDashboard": None,
        "ExplanationDashboard": None,
        "ErrorAnalysisDashboard": None,
        "ModelAnalyzer": None,
        "TabularExplainer": None,
        "MetricFrame": None,
        "selection_rate": None,
        "true_positive_rate": None,
        "false_positive_rate": None,
        "false_negative_rate": None,
    }

    # --- responsibleai + raiwidgets ---
    try:
        from responsibleai import RAIInsights
        from responsibleai.feature_metadata import FeatureMetadata
        from raiwidgets import (
            ResponsibleAIDashboard,
            ExplanationDashboard,
            ErrorAnalysisDashboard,
        )
        status["_RAI"] = True
        status["_ERRANALYSIS"] = True
        objects.update({
            "RAIInsights": RAIInsights,
            "FeatureMetadata": FeatureMetadata,
            "ResponsibleAIDashboard": ResponsibleAIDashboard,
            "ExplanationDashboard": ExplanationDashboard,
            "ErrorAnalysisDashboard": ErrorAnalysisDashboard,
        })
    except Exception as exc:
        print(f"RAI core/widgets unavailable: {exc}")

    # --- interpret-community ---
    try:
        from interpret_community.tabular_explainer import TabularExplainer
        status["_INTERPRET"] = True
        objects["TabularExplainer"] = TabularExplainer
    except Exception as exc:
        print(f"Interpret-Community unavailable: {exc}")

    # --- erroranalysis ---
    try:
        from erroranalysis import ModelAnalyzer
        status["_ERRANALYSIS"] = True
        objects["ModelAnalyzer"] = ModelAnalyzer
    except Exception as exc:
        if not status["_ERRANALYSIS"]:
            print(f"Error Analysis unavailable: {exc}")

    # --- fairlearn ---
    try:
        from fairlearn.metrics import (
            MetricFrame,
            selection_rate,
            true_positive_rate,
            false_positive_rate,
            false_negative_rate,
        )
    except Exception:
        try:
            from fairlearn.metrics import (
                MetricFrame,
                selection_rate,
                true_positive_rate,
                false_positive_rate,
            )

            def false_negative_rate(y_true, y_pred):
                tn, fp, fn, tp = confusion_matrix(
                    y_true, y_pred).ravel()
                return fn / (fn + tp) if (fn + tp) > 0 else 0.0

            status["_FAIRLEARN"] = True
            objects.update({
                "MetricFrame": MetricFrame,
                "selection_rate": selection_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            })
        except Exception as inner_exc:
            print(f"Fairlearn metrics unavailable: {inner_exc}")
    else:
        status["_FAIRLEARN"] = True
        objects.update({
            "MetricFrame": MetricFrame,
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        })

    return status, objects


# ──────────────────────────────────────────────
#  Score extraction
# ──────────────────────────────────────────────

def positive_scores(estimator, X) -> np.ndarray:
    """Return continuous scores for the positive class (class 1)."""
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            classes = list(getattr(estimator, "classes_", [0, 1]))
            pos_idx = classes.index(1) if 1 in classes else 1
            return proba[:, pos_idx].ravel()
        return proba.ravel()
    if hasattr(estimator, "decision_function"):
        df = np.asarray(estimator.decision_function(X))
        if df.ndim == 2 and df.shape[1] >= 2:
            classes = list(getattr(estimator, "classes_", [0, 1]))
            pos_idx = classes.index(1) if 1 in classes else 1
            return df[:, pos_idx].ravel()
        return df.ravel()
    raise AttributeError(
        "Estimator must implement predict_proba or decision_function"
    )


# ──────────────────────────────────────────────
#  AUC reporting
# ──────────────────────────────────────────────

def auc_report(y_true, y_score, name: str = "model",
               plot: bool = True) -> Dict[str, float]:
    """Print and return ROC AUC, PR AUC, prevalence, and lift."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    pr_auc = float(average_precision_score(y_true, y_score))
    roc = float(roc_auc_score(y_true, y_score))
    prevalence = float(np.mean(y_true))
    lift = float(pr_auc / prevalence) if prevalence > 0 else float("inf")

    print(f"{name}")
    print(f"  PR AUC      : {pr_auc:.3f}")
    print(f"  ROC AUC     : {roc:.3f}")
    print(f"  Prevalence  : {prevalence:.3f}")
    print(f"  PR AUC / prevalence (lift): {lift:.2f}×")

    if plot:
        # ROC curve
        roc_disp = RocCurveDisplay.from_predictions(y_true, y_score)
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        roc_disp.line_.set_label(f"Model (ROC AUC = {roc:.3f})")
        roc_disp.ax_.set_xlabel("False positive rate")
        roc_disp.ax_.set_ylabel("True positive rate (recall)")
        roc_disp.ax_.legend()
        plt.title(f"ROC curve — {name} (AUC = {roc:.3f})")
        plt.show()

        # Precision-Recall curve
        pr_disp = PrecisionRecallDisplay.from_predictions(y_true, y_score)
        plt.hlines(prevalence, 0, 1, colors="gray", linestyles="dotted",
                   label=f"Prevalence = {prevalence:.3f}")
        pr_disp.line_.set_label(f"Model (PR AUC = {pr_auc:.3f})")
        pr_disp.ax_.set_xlabel("Recall (sensitivity)")
        pr_disp.ax_.set_ylabel("Precision (positive predictive value)")
        pr_disp.ax_.legend()
        plt.title(f"Precision-Recall curve — {name} (PR AUC = {pr_auc:.3f})")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    return {"name": name, "roc_auc": roc, "pr_auc": pr_auc,
            "prevalence": prevalence, "lift": lift}


# ──────────────────────────────────────────────
#  Threshold trade-off table
# ──────────────────────────────────────────────

def tradeoff_table(y_true, y_score,
                   thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Precision, recall, confusion counts and alert rates at many thresholds."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel()

    if not np.isfinite(y_score).all():
        raise ValueError("y_score must contain only finite values")

    if thresholds is None:
        thresholds = np.unique(np.concatenate(([0.0], y_score, [1.0])))
    else:
        thresholds = np.asarray(thresholds, dtype=float).ravel()

    rows = []
    n = len(y_true)
    for t in thresholds:
        y_hat = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "alerts_per_1000": 1000.0 * float(np.mean(y_hat)),
            "true_pos_per_1000": 1000.0 * float(tp) / n,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
#  Threshold selection policies
# ──────────────────────────────────────────────

def pick_threshold_cost(y_true, y_score, C_FP: float, C_FN: float,
                        thresholds: Optional[np.ndarray] = None
                        ) -> Dict[str, object]:
    """Select threshold minimising illustrative FP/FN cost."""
    tbl = tradeoff_table(y_true, y_score, thresholds)
    denom = C_FP + C_FN
    t_formula = float(C_FP / denom) if denom > 0 else 1.0

    tv = np.asarray(tbl["threshold"].values, dtype=float)
    idx_near = int(np.argmin(np.abs(tv - t_formula)))
    row_formula = tbl.iloc[idx_near].copy()

    exp_cost = C_FP * tbl["FP"] + C_FN * tbl["FN"]
    idx_emp = int(exp_cost.values.argmin())
    row_emp = tbl.iloc[idx_emp].copy()

    summary = pd.DataFrame([
        {"rule": "Bayes formula", **row_formula.to_dict(),
         "expected_cost": float(exp_cost.iloc[idx_near])},
        {"rule": "Empirical min cost", **row_emp.to_dict(),
         "expected_cost": float(exp_cost.iloc[idx_emp])},
    ])
    return {
        "threshold_formula": float(row_formula["threshold"]),
        "threshold_empirical": float(row_emp["threshold"]),
        "summary": summary,
        "table": tbl,
    }


def pick_threshold_recall_floor(y_true, y_score, recall_floor: float,
                                thresholds: Optional[np.ndarray] = None
                                ) -> Dict[str, object]:
    """Max precision subject to recall ≥ recall_floor."""
    tbl = tradeoff_table(y_true, y_score, thresholds)
    feasible = tbl[tbl["recall"] >= recall_floor]
    if len(feasible) == 0:
        idx = int(np.argmax(tbl["recall"].to_numpy()))
        chosen = tbl.iloc[idx]
        rule = "Max recall fallback"
    else:
        max_prec = feasible["precision"].max()
        candidates = feasible[feasible["precision"] == max_prec]
        idx = int(candidates["threshold"].values.argmax())
        chosen = candidates.iloc[idx]
        rule = "Recall floor then max precision"
    summary = pd.DataFrame([{"rule": rule, **chosen.to_dict()}])
    return {"threshold": float(chosen["threshold"]),
            "summary": summary, "table": tbl}


def pick_threshold_workload(y_true, y_score, alerts_per_1000_max: float,
                            thresholds: Optional[np.ndarray] = None
                            ) -> Dict[str, object]:
    """Max TP/1 000 under alerts-per-1 000 budget."""
    tbl = tradeoff_table(y_true, y_score, thresholds)
    feasible = tbl[tbl["alerts_per_1000"] <= alerts_per_1000_max + 1e-9]
    if len(feasible) == 0:
        idx = int((tbl["alerts_per_1000"] -
                   alerts_per_1000_max).abs().values.argmin())
        chosen = tbl.iloc[idx]
        rule = "Closest to alerts budget fallback"
    else:
        best_tp = feasible["true_pos_per_1000"].max()
        candidates = feasible[feasible["true_pos_per_1000"] == best_tp]
        best_prec = candidates["precision"].max()
        candidates = candidates[candidates["precision"] == best_prec]
        idx = int(candidates["threshold"].values.argmax())
        chosen = candidates.iloc[idx]
        rule = "Max TP per 1000 under budget"
    summary = pd.DataFrame([{"rule": rule, **chosen.to_dict()}])
    return {"threshold": float(chosen["threshold"]),
            "summary": summary, "table": tbl}


# ──────────────────────────────────────────────
#  Single-threshold summary
# ──────────────────────────────────────────────

def summary_at_threshold(y_true, y_score, threshold) -> pd.DataFrame:
    """One-row summary at a specific threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    y_hat = (y_score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    n = len(y_true)
    return pd.DataFrame([{
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
        "alerts_per_1000": 1000.0 * float(np.mean(y_hat)),
        "true_pos_per_1000": 1000.0 * float(tp) / n,
    }])

# ──────────────────────────────────────────────
#  Subgroup Report
# ──────────────────────────────────────────────

def subgroup_report(y_true, y_score, groups, group_map, threshold=0.5):
    """Subgroup performance at a fixed threshold, with Wilson 95% CIs on recall.

    Parameters
    ----------
    y_true, y_score : array-like  (labels, positive-class scores)
    groups : array-like           subgroup membership vector
    group_map : dict              {display_label: subgroup_value}
    threshold : float

    Returns a DataFrame with N, failures, prevalence, recall (+CI), precision,
    missed cases (FN), and alerts per 1,000.  Groups with a single class get
    NaN recall/precision rather than a misleading number.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel()
    groups = np.asarray(groups).ravel()
    y_pred = (y_score >= float(threshold)).astype(int)

    rows = []
    for label, value in group_map.items():
        mask = groups == value
        n = int(mask.sum())
        if n == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        failures = int(yt.sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = failures - tp
        row = {"Group": label, "N": n, "Failures": failures,
               "Prevalence": round(failures / n, 3)}
        if 0 < failures < n:
            lo, hi = wilson_interval(tp, failures)
            row.update({"Recall": round(tp / failures, 3),
                        "Recall 95% CI": f"{lo:.2f}-{hi:.2f}",
                        "Precision": round(precision_score(yt, yp, zero_division=0), 3)})
        else:
            row.update({"Recall": np.nan, "Recall 95% CI": "n/a",
                        "Precision": np.nan})
        row.update({"Missed (FN)": fn, "Alerts/1000": round(yp.mean() * 1000)})
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
#  Validation visualisations
# ──────────────────────────────────────────────

def plot_recall_floor_curves(y_true, y_score, recall_floor, chosen_threshold):
    """Precision and recall vs threshold with recall floor and chosen threshold."""
    tbl = tradeoff_table(y_true, y_score)
    chosen = summary_at_threshold(y_true, y_score, chosen_threshold).iloc[0]

    plt.figure()
    plt.plot(tbl["threshold"], tbl["recall"], label="Recall")
    plt.plot(tbl["threshold"], tbl["precision"], label="Precision")
    plt.axhline(float(recall_floor), linestyle="--", color="red",
                label=f"Recall floor = {float(recall_floor):.2f}")
    plt.axvline(float(chosen_threshold), linestyle=":", color="black",
                label=f"Chosen threshold = {float(chosen_threshold):.2f}")

    plt.scatter(float(chosen_threshold), chosen["recall"],
                color="blue", zorder=5)
    plt.text(float(chosen_threshold) + 0.01, chosen["recall"],
             f"Recall={chosen['recall']:.2f}", va="center")
    plt.scatter(float(chosen_threshold), chosen["precision"],
                color="orange", zorder=5)
    plt.text(float(chosen_threshold) + 0.01, chosen["precision"],
             f"Prec={chosen['precision']:.2f}", va="center")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Recall floor then maximize precision")
    plt.legend()
    plt.xlim(0, 0.55)
    plt.show()


def plot_cumulative_recall_at_threshold(y_true, y_score, chosen_threshold):
    """Cumulative recall vs number of alerts."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    total_pos = int(cum_tp[-1]) if cum_tp.size else 0
    alerts = np.arange(1, len(y_sorted) + 1)
    recall_curve = (cum_tp / total_pos if total_pos > 0
                    else np.zeros_like(cum_tp, dtype=float))

    y_hat = (y_score >= float(chosen_threshold)).astype(int)
    n_alerts = int(y_hat.sum())
    rec_at = (float(recall_curve[n_alerts - 1])
              if 0 < n_alerts <= len(y_sorted) else 0.0)

    plt.figure()
    plt.plot(alerts, recall_curve, label="Cumulative recall")
    plt.axvline(n_alerts, linestyle="--", color="red",
                label=f"Alerts = {n_alerts}")
    plt.scatter(n_alerts, rec_at, color="black", zorder=5)
    plt.text(n_alerts + max(2, len(y_sorted) // 100), rec_at,
             f"Recall = {rec_at:.2f}", va="center")
    plt.xlabel("Number of alerts")
    plt.ylabel("Cumulative recall (true cases captured)")
    plt.title("Cumulative capture of true cases vs alerts")
    plt.legend()
    plt.show()


def plot_topk_at_threshold(y_true, y_score, chosen_threshold, top_k=30):
    """Bar chart of top-k highest-risk cases coloured by true label."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()

    order = np.argsort(-y_score)
    top_idx = order[:int(top_k)]
    top_scores = y_score[top_idx]
    top_true = y_true[top_idx]

    tp_mask = np.where(top_true == 1)[0]
    fp_mask = np.where(top_true == 0)[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(tp_mask, top_scores[tp_mask], color="tab:red",
           label="True positive (correctly flagged)")
    ax.bar(fp_mask, top_scores[fp_mask], color="tab:gray",
           label="False positive (false alert)")
    ax.axhline(float(chosen_threshold), linestyle="--", color="black",
               label=f"Threshold = {float(chosen_threshold):.2f}")
    ax.set_xlabel("Patients ranked by predicted risk")
    ax.set_ylabel("Predicted risk score")
    ax.set_title(f"Top {int(top_k)} highest-risk patients on validation")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
#  Thresholded estimator wrapper (for RAI dashboard)
# ──────────────────────────────────────────────

class ThresholdedEstimator(BaseEstimator, ClassifierMixin):
    """Wrap a probabilistic classifier so `.predict` uses a custom threshold.

    `.predict_proba` is passed through unchanged so RAI tools
    can still inspect calibrated probabilities.
    """

    def __init__(self, base, threshold: float = 0.5, positive_label=1):
        self.base = base
        self.threshold = float(threshold)
        self.positive_label = positive_label

    def fit(self, X, y=None, **fit_params):
        if hasattr(self.base, "fit") and y is not None:
            self.base.fit(X, y, **fit_params)
        self.classes_ = getattr(self.base, "classes_", None)
        self.n_features_in_ = getattr(self.base, "n_features_in_", None)
        self.feature_names_in_ = getattr(
            self.base, "feature_names_in_", None)
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        if not hasattr(self.base, "predict_proba"):
            raise AttributeError(
                f"{type(self.base).__name__} lacks predict_proba")
        return self._ensure_2d(self.base.predict_proba(X))

    def predict(self, X):
        p_pos = self._pos_col(self.predict_proba(X))
        if getattr(self, "classes_", None) is not None and len(self.classes_) == 2:
            neg = [c for c in self.classes_ if c != self.positive_label][0]
            return np.where(p_pos >= self.threshold,
                            self.positive_label, neg)
        return (p_pos >= self.threshold).astype(int)

    def decision_function(self, X):
        if hasattr(self.base, "decision_function"):
            return self.base.decision_function(X)
        return self._pos_col(self.predict_proba(X)) - self.threshold

    def predict_log_proba(self, X):
        if hasattr(self.base, "predict_log_proba"):
            return self._ensure_2d(self.base.predict_log_proba(X))
        return np.log(np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15))

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)
        return self

    # --- helpers ---
    @staticmethod
    def _ensure_2d(arr):
        arr = np.asarray(arr)
        return np.c_[1 - arr, arr] if arr.ndim == 1 else arr

    def _pos_col(self, proba_2d):
        proba_2d = np.asarray(proba_2d)
        if proba_2d.ndim == 1:
            return proba_2d
        if getattr(self, "classes_", None) is not None:
            classes = list(self.classes_)
            if self.positive_label in classes:
                return proba_2d[:, classes.index(self.positive_label)]
        return proba_2d[:, min(1, proba_2d.shape[1] - 1)]

    def __getattr__(self, name):
        return getattr(self.base, name)


def make_thresholded_estimator(base_estimator, threshold: float = 0.5,
                               positive_label=1):
    """Convenience factory for ThresholdedEstimator."""
    return ThresholdedEstimator(base_estimator, threshold=threshold,
                               positive_label=positive_label)


# ──────────────────────────────────────────────
#  Additions: calibration, uncertainty, leakage audit, fairness helpers
# ──────────────────────────────────────────────
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-frequency-bin Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true).ravel(); y_prob = np.asarray(y_prob).ravel()
    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for idx in bins:
        if len(idx):
            ece += len(idx) / len(y_true) * abs(y_true[idx].mean() - y_prob[idx].mean())
    return float(ece)


def calibration_summary(y_true, scores: dict) -> pd.DataFrame:
    """Brier, Brier skill score (vs. prevalence-only forecast), ECE, CITL."""
    y_true = np.asarray(y_true).ravel()
    p = y_true.mean()
    ref = brier_score_loss(y_true, np.full(len(y_true), p))
    rows = []
    for name, s_ in scores.items():
        b = brier_score_loss(y_true, s_)
        rows.append({"Model": name, "Brier": round(b, 4),
                     "Brier skill vs prevalence": round(1 - b / ref, 3),
                     "ECE": round(expected_calibration_error(y_true, s_), 4),
                     "Mean pred": round(float(np.mean(s_)), 3),
                     "Observed": round(p, 3)})
    return pd.DataFrame(rows)


def plot_reliability(y_true, scores: dict, n_bins: int = 8, title="Reliability plot", save=None):
    """Reliability curves (equal-frequency bins) + histogram of predictions."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 7), height_ratios=[3, 1],
                                  sharex=True)
    for name, s_ in scores.items():
        fp, mp = calibration_curve(y_true, s_, n_bins=n_bins, strategy="quantile")
        ax.plot(mp, fp, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_ylabel("Observed failure rate"); ax.set_title(title); ax.legend()
    for name, s_ in scores.items():
        ax2.hist(s_, bins=20, alpha=.5, label=name)
    ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Patients")
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def bootstrap_metric_ci(y_true, y_score, fn, n_boot: int = 1000, seed: int = 42):
    """Percentile bootstrap CI for any fn(y, score) -> float."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        vals.append(fn(y_true[i], y_score[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def threshold_stability(y_true, y_score, threshold, n_boot: int = 2000, seed: int = 42):
    """Bootstrap the recall / alerts-per-1000 obtained at a FIXED threshold.

    Shows how much a threshold tuned on ~78 positives can move on new patients.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    rec, alr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[i], y_score[i]
        pred = ys >= threshold
        if yt.sum() == 0:
            continue
        rec.append((pred & (yt == 1)).sum() / yt.sum()); alr.append(pred.mean() * 1000)
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"recall_ci": q(rec), "alerts_ci": q(alr)}


def tiered_alert_table(y_true, y_score, t_high, t_low, label=""):
    """Two-tier operating point: Tier 1 (score>=t_high) gets intensive follow-up,
    Tier 2 (t_low<=score<t_high) gets a light-touch check. Returns one row."""
    y = np.asarray(y_true).astype(int); s_ = np.asarray(y_score)
    t1 = s_ >= t_high; t2 = (s_ >= t_low) & ~t1
    pos = max(y.sum(), 1)
    return {"Set": label,
            "Tier1 alerts/1000": round(t1.mean() * 1000),
            "Tier2 alerts/1000": round(t2.mean() * 1000),
            "Recall Tier1": round((t1 & (y == 1)).sum() / pos, 3),
            "Recall Tier1+2": round(((t1 | t2) & (y == 1)).sum() / pos, 3),
            "Missed (untiered)": int(((~(t1 | t2)) & (y == 1)).sum())}


def group_recall_floor_thresholds(y_true, y_score, groups, recall_floor=0.60):
    """Per-group thresholds: the highest threshold in each group that still gives
    recall >= floor on the FITTING data (validation). Returns {group_value: thr}."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        pos_scores = np.sort(y_score[m][y_true[m] == 1])[::-1]
        k = int(np.ceil(recall_floor * len(pos_scores)))
        out[g] = float(pos_scores[k - 1])
    return out


def apply_group_thresholds(y_score, groups, thr_map):
    y_score = np.asarray(y_score); groups = np.asarray(groups)
    thr = np.array([thr_map[g] for g in groups])
    return (y_score >= thr).astype(int)


def ppv_under_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """PPV if the SAME classifier (TPR/FPR) were used at a different prevalence."""
    return tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))


def leakage_audit(X_train, X_val, X_test, forbidden, fitted_pipeline, num_cols):
    """Computed (not hard-coded) leakage checks. Returns a DataFrame."""
    checks = []
    feats = set(X_train.columns)
    checks.append(("No post-randomisation / outcome-derived columns in features",
                   not (feats & set(forbidden)), f"forbidden={sorted(forbidden)}"))
    idx = [set(X_train.index), set(X_val.index), set(X_test.index)]
    checks.append(("Train / val / test are disjoint patients",
                   not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2]),
                   f"sizes={[len(i) for i in idx]}"))
    scaler = fitted_pipeline.named_steps["prep"].named_transformers_["num"]
    checks.append(("StandardScaler statistics come from TRAIN only",
                   np.allclose(scaler.mean_, X_train[num_cols].mean().values),
                   "scaler.mean_ == train mean (differs from val/test mean)"))
    checks.append(("Scaler mean differs from full-data mean (proves no peeking)",
                   not np.allclose(scaler.mean_, np.concatenate([X_train[num_cols].values,
                                    X_val[num_cols].values, X_test[num_cols].values]).mean(0)),
                   ""))
    return pd.DataFrame(checks, columns=["Check", "Passed", "Evidence"])


# ──────────────────────────────────────────────
#  RAI-result extraction helpers (static equivalents of dashboard views)
# ──────────────────────────────────────────────

def subgroup_auc(y_true, y_score, groups, group_map, n_boot=500, seed=42):
    """ROC AUC per subgroup with bootstrap 95% CI."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); groups = np.asarray(groups)
    rows = []
    for label, value in group_map.items():
        m = groups == value
        yt, ys = y_true[m], y_score[m]
        if len(np.unique(yt)) < 2:
            rows.append({"Group": label, "N": int(m.sum()), "Failures": int(yt.sum()),
                         "ROC AUC": np.nan, "95% CI": "n/a"}); continue
        lo, hi = bootstrap_metric_ci(yt, ys, roc_auc_score, n_boot=n_boot, seed=seed)
        rows.append({"Group": label, "N": int(m.sum()), "Failures": int(yt.sum()),
                     "ROC AUC": round(roc_auc_score(yt, ys), 3), "95% CI": f"{lo:.2f}-{hi:.2f}"})
    return pd.DataFrame(rows)


def error_tree_paths(error_report, min_size: int = 15, top: int = 6) -> pd.DataFrame:
    """Turn the RAI error-analysis tree into readable leaf rules, worst first."""
    nodes = {n["id"]: n for n in error_report.tree}
    children = {}
    for n in error_report.tree:
        children.setdefault(n["parentId"], []).append(n["id"])
    rows = []
    for nid, n in nodes.items():
        if nid in children or n["parentId"] is None:
            continue  # only leaves
        conds, cur = [], n
        while cur["parentId"] is not None:
            conds.append(cur["condition"]); cur = nodes[cur["parentId"]]
        size = int(n["size"])
        if size >= min_size:
            rows.append({"Rule (path from root)": " AND ".join(reversed(conds)),
                         "Patients": size, "Errors": int(n["error"]),
                         "Error rate": round(n["error"] / size, 3)})
    return pd.DataFrame(rows).sort_values("Error rate", ascending=False).head(top).reset_index(drop=True)


def whatif_risk_by_arm(model, X, arm_col="arm", arms=None):
    """Predicted risk for every patient under every regimen (all else equal)."""
    arms = arms or list(X[arm_col].unique())
    out = {}
    for a in arms:
        Xa = X.copy(); Xa[arm_col] = a
        out[a] = positive_scores(model, Xa)
    return pd.DataFrame(out, index=X.index)


def cf_summary(cf_result, pred_col_name="test_pred"):
    """Summarise DiCE counterfactuals restricted to the treatment column."""
    rows = []
    for ex in cf_result.cf_examples_list:
        orig = ex.test_instance_df.iloc[0]
        cfs = ex.final_cfs_df
        pred0 = int(np.ravel(ex.test_pred)[0]) if np.ndim(ex.test_pred) else int(ex.test_pred)
        rows.append({"orig_arm": orig["arm"], "orig_pred": pred0,
                     "n_flip_arms": 0 if cfs is None else len(cfs),
                     "flip_arms": [] if cfs is None else list(cfs["arm"])})
    return pd.DataFrame(rows)


def monitoring_check(y_true, y_score, threshold, race, gender,
                     alert_cap=300, recall_floor_group=0.50, fnr_gap_max=0.10,
                     calib_gap_max=0.05):
    """One monitoring cycle: compute the KPIs and compare with alert triggers.

    Designed to be run monthly/quarterly on the latest cohort with known outcomes.
    """
    y = np.asarray(y_true).astype(int); s_ = np.asarray(y_score)
    race = np.asarray(race); gender = np.asarray(gender)
    pred = (s_ >= threshold).astype(int)
    def rec(m):
        return np.nan if y[m].sum() == 0 else (pred[m] & y[m]).sum() / y[m].sum()
    r = {g: rec(race == g) for g in (0, 1)}; gd = {g: rec(gender == g) for g in (0, 1)}
    rows = [
        ("Alerts per 1,000", round(pred.mean() * 1000), f"<= {alert_cap}", pred.mean() * 1000 <= alert_cap),
        ("Recall White", round(r[0], 3), f">= {recall_floor_group}", r[0] >= recall_floor_group),
        ("Recall Non-white", round(r[1], 3), f">= {recall_floor_group}", r[1] >= recall_floor_group),
        ("Recall Female", round(gd[0], 3), f">= {recall_floor_group}", gd[0] >= recall_floor_group),
        ("Recall Male", round(gd[1], 3), f">= {recall_floor_group}", gd[1] >= recall_floor_group),
        ("FNR gap (race)", round(abs(r[0] - r[1]), 3), f"<= {fnr_gap_max}", abs(r[0] - r[1]) <= fnr_gap_max),
        ("|mean pred - observed|", round(abs(s_.mean() - y.mean()), 3), f"<= {calib_gap_max}", abs(s_.mean() - y.mean()) <= calib_gap_max),
    ]
    out = pd.DataFrame(rows, columns=["KPI", "Value", "Trigger (OK if)", "OK"])
    out["Status"] = np.where(out["OK"], "OK", "ALERT")
    return out.drop(columns="OK")


class NumericArmAdapter(BaseEstimator, ClassifierMixin):
    """Lets libraries that force float inputs (e.g. Fairlearn's ThresholdOptimizer)
    call a model that expects a string treatment column.

    The arm column is passed as an integer code and decoded back to its label
    before the wrapped model sees it. Used ONLY for post-processing experiments."""

    def __init__(self, base, columns, arm_col, arm_labels):
        self.base, self.columns, self.arm_col, self.arm_labels = base, list(columns), arm_col, list(arm_labels)
        self.classes_ = getattr(base, 'classes_', np.array([0, 1]))   # marks the adapter as 'fitted'

    def fit(self, X=None, y=None, **kw):   # no-op: the wrapped model is already fitted
        return self

    def _frame(self, X):
        df_ = pd.DataFrame(np.asarray(X, dtype=float), columns=self.columns)
        df_[self.arm_col] = [self.arm_labels[int(round(v))] for v in df_[self.arm_col]]
        return df_

    def predict_proba(self, X):
        return self.base.predict_proba(self._frame(X))

    def predict(self, X):
        return self.base.predict(self._frame(X))


def encode_arm(X, arm_col, arm_labels):
    """Numeric copy of X with the treatment label replaced by its integer code."""
    X = X.copy(); X[arm_col] = X[arm_col].map({a: i for i, a in enumerate(arm_labels)}); return X
