#!/usr/bin/env python3
"""
logreg_churn_end_to_end.py

End-to-end scikit-learn Logistic Regression example for Customer Success teams.
Predicts churn probability so CSMs can proactively re-engage customers.

Run:
  python logreg_churn_end_to_end.py

Notes:
- Saves a single Joblib artifact that includes the fitted pipeline + feature names + schema + metrics.
- Includes robust inspection utilities to view what's inside a .joblib file safely (from a code standpoint).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED: int = 42
ARTIFACT_PATH: Path = Path("churn_logreg_pipeline.joblib")


@dataclass(frozen=True)
class Artifact:
    """A structured payload we persist to joblib for future inference + auditability."""
    pipeline: Pipeline
    feature_names: np.ndarray  # 1D array of output feature names after preprocessing
    input_schema: Dict[str, str]  # column -> dtype
    metrics: Dict[str, Any]
    created_at_utc: str
    version: str = "1.0.0"


def make_sample_data(n: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Make a sample dataset of n rows.

    Each row represents a customer, with features relevant to churn prediction.
    Target variable: churned (1 if churned, 0 otherwise).

    Parameters
    ----------
    n : int
        Number of rows to generate.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(0, 72, n),
            "monthly_charges": np.clip(rng.normal(75, 25, n), 20, 200),
            "support_tickets_90d": rng.poisson(1.3, n),
            "late_payments_6m": rng.poisson(0.6, n),
            "autopay": rng.choice(["yes", "no"], p=[0.55, 0.45], size=n),
            "contract_type": rng.choice(
                ["month-to-month", "one-year", "two-year"], p=[0.6, 0.25, 0.15], size=n
            ),
            "plan_tier": rng.choice(["basic", "standard", "premium"], p=[0.5, 0.35, 0.15], size=n),
        }
    )

    logits = (
        -2.0
        + 0.02 * (df["monthly_charges"] - 70)
        + 0.4 * df["support_tickets_90d"]
        + 0.6 * df["late_payments_6m"]
        - 0.03 * df["tenure_months"]
        + df["autopay"].map({"yes": -0.6, "no": 0.0}).astype(float)
        + df["contract_type"].map(
            {"month-to-month": 0.9, "one-year": -0.3, "two-year": -0.8}
        ).astype(float)
        + df["plan_tier"].map({"basic": 0.2, "standard": 0.0, "premium": -0.2}).astype(float)
    )

    p = 1.0 / (1.0 + np.exp(-logits))
    df["churned"] = (rng.random(n) < p).astype(int)
    return df


def threshold_policy(p: float) -> str:
    """
    Map churn probability to a recommended CSM action.
    """
    if p >= 0.8:
        return "Immediate CSM outreach + save offer"
    if p >= 0.6:
        return "Proactive engagement"
    if p >= 0.4:
        return "Monitor + nudge"
    return "No action"


def build_pipeline(
    num_features: List[str],
    cat_features: List[str],
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """
    Build the preprocessing + LogisticRegression pipeline.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Dense=False is default; leaving as sparse output is fine for LR in sklearn
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,  # makes names like "num__tenure_months"
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        n_jobs=None,  # LR doesn't always use n_jobs depending on solver; keep default
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def evaluate_binary_classifier(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute standard evaluation metrics for a probabilistic binary classifier.
    """
    preds = (probs >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "classification_report": classification_report(y_true, preds, output_dict=True),
    }
    return metrics


def get_feature_names(pipe: Pipeline) -> np.ndarray:
    """
    Extract output feature names from the preprocessing step.

    Works for:
    - Pipeline with a ColumnTransformer step named "prep" (as built above)
    """
    prep = pipe.named_steps.get("prep")
    if prep is None:
        raise ValueError("Pipeline has no 'prep' step; cannot extract feature names.")

    if not hasattr(prep, "get_feature_names_out"):
        raise ValueError("Preprocessor does not support get_feature_names_out().")

    return prep.get_feature_names_out()


def save_artifact(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    metrics: Dict[str, Any],
    path: Path = ARTIFACT_PATH,
) -> None:
    """
    Save pipeline + feature names + schema + metrics as a single joblib artifact.
    """
    artifact = Artifact(
        pipeline=pipe,
        feature_names=get_feature_names(pipe),
        input_schema={c: str(t) for c, t in X_train.dtypes.items()},
        metrics=metrics,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    dump(asdict(artifact), path)
    print(f"\nSaved artifact: {path.resolve()}")


def load_and_inspect_joblib(path: Path = ARTIFACT_PATH, preview_n: int = 30) -> None:
    """
    Load a joblib file and print a high-signal inspection of contents.
    """
    obj = joblib.load(path)
    print("\nLoaded object type:", type(obj))

    # If we saved the Artifact as dict (recommended above)
    if isinstance(obj, dict):
        print("\nTop-level keys:", list(obj.keys()))

        pipe = obj.get("pipeline")
        if isinstance(pipe, Pipeline):
            print("\nPipeline:")
            print(pipe)
            print("\nPipeline steps:", list(pipe.named_steps.keys()))

        fns = obj.get("feature_names")
        if isinstance(fns, (list, np.ndarray)):
            fns_arr = np.array(fns)
            print("\nFeature names count:", len(fns_arr))
            print(f"First {min(preview_n, len(fns_arr))} feature names:", fns_arr[:preview_n])

        metrics = obj.get("metrics")
        if isinstance(metrics, dict):
            print("\nMetrics summary:")
            # keep output compact
            for k in ["roc_auc", "pr_auc", "threshold"]:
                if k in metrics:
                    print(f"  {k}: {metrics[k]}")
        return

    # Back-compat: if user saved only the pipeline
    if isinstance(obj, Pipeline):
        print("\nPipeline:")
        print(obj)
        print("\nPipeline steps:", list(obj.named_steps.keys()))
        try:
            fns = get_feature_names(obj)
            print("\nFeature names count:", len(fns))
            print(f"First {min(preview_n, len(fns))} feature names:", fns[:preview_n])
        except Exception as e:
            print("\nCould not extract feature names:", e)
        return

    # Generic fallback
    if hasattr(obj, "__dict__"):
        print("\nObject __dict__ keys:", list(obj.__dict__.keys())[:50])
    else:
        print("\nObject has no __dict__ (likely a built-in / numpy type).")


def main() -> None:
    df = make_sample_data()
    X = df.drop(columns=["churned"])
    y = df["churned"].to_numpy()

    num_features = ["tenure_months", "monthly_charges", "support_tickets_90d", "late_payments_6m"]
    cat_features = ["autopay", "contract_type", "plan_tier"]

    pipe = build_pipeline(num_features=num_features, cat_features=cat_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_classifier(y_true=y_test, probs=probs, threshold=0.5)

    print("ROC AUC:", round(metrics["roc_auc"], 3))
    print("PR AUC :", round(metrics["pr_auc"], 3))
    print("Confusion Matrix:\n", np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:\n", classification_report(y_test, (probs >= 0.5).astype(int)))

    # Sample decisions
    sample = X_test.head(8).copy()
    sample_probs = pipe.predict_proba(sample)[:, 1]
    sample["churn_probability"] = sample_probs
    sample["recommended_action"] = [threshold_policy(p) for p in sample_probs]

    print("\nSample CSM Decisions:")
    print(sample)

    # Save full artifact (pipeline + feature names + schema + metrics)
    save_artifact(pipe=pipe, X_train=X_train, metrics=metrics, path=ARTIFACT_PATH)

    # Inspect what we saved
    load_and_inspect_joblib(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
