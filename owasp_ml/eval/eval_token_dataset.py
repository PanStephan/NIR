from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


PKG_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = PKG_DIR / "prep"
REPORTS_DIR = PKG_DIR


def _load_token_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or (DATASETS_DIR / "token_risk_dataset.csv")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["risk_type", "risk_label"])
    df["risk_type"] = df["risk_type"].astype(str)
    df["risk_label"] = df["risk_label"].astype(int)
    return df


def _compute_sanity_stats(df: pd.DataFrame) -> Dict[str, object]:
    total_samples = int(len(df))
    label_counts = df["risk_label"].value_counts().sort_index()
    label_share = (label_counts / total_samples).to_dict()
    owasp_counts = df["owasp_category"].value_counts(dropna=False)
    owasp_share = (owasp_counts / total_samples).to_dict()
    chain_counts = df["chain"].value_counts(dropna=False)
    entity_counts = df["entity_type"].value_counts(dropna=False)
    loss_stats = df["loss_usd"].describe().to_dict()
    return {
        "total_samples": total_samples,
        "label_counts": {int(k): int(v) for k, v in label_counts.to_dict().items()},
        "label_share": {int(k): float(v) for k, v in label_share.items()},
        "owasp_counts": {str(k): int(v) for k, v in owasp_counts.to_dict().items()},
        "owasp_share": {str(k): float(v) for k, v in owasp_share.items()},
        "chain_counts": {str(k): int(v) for k, v in chain_counts.to_dict().items()},
        "entity_counts": {str(k): int(v) for k, v in entity_counts.to_dict().items()},
        "loss_stats": loss_stats,
    }


def _train_test_baseline(df: pd.DataFrame) -> Tuple[Dict[str, float], str]:
    X = df["risk_type"].tolist()
    y = df["risk_label"].astype(int).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = LogisticRegression(
        max_iter=1000,
    )
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report_text = classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0,
    )
    proba = classifier.predict_proba(X_test_vec)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    metrics = {
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
    }
    return metrics, report_text


def _kfold_cv(df: pd.DataFrame) -> Dict[str, float]:
    X = df["risk_type"].tolist()
    y = df["risk_label"].astype(int).to_numpy()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
    )
    X_vec = vectorizer.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores: List[float] = []
    auc_scores: List[float] = []
    for train_idx, test_idx in skf.split(X_vec, y):
        X_train = X_vec[train_idx]
        X_test = X_vec[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf = LogisticRegression(
            max_iter=1000,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1_scores.append(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )
        proba = clf.predict_proba(X_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, proba))
    return {
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "roc_auc_mean": float(np.mean(auc_scores)),
        "roc_auc_std": float(np.std(auc_scores)),
    }


def _format_report(
    sanity: Dict[str, object],
    baseline_metrics: Dict[str, float],
    baseline_report: str,
    cv_metrics: Dict[str, float],
) -> str:
    lines: List[str] = []
    lines.append("=== Sanity-проверка token_risk_dataset ===")
    lines.append(f"Total samples: {sanity['total_samples']}")
    lines.append("Risk_label counts:")
    for label, count in sanity["label_counts"].items():
        share = sanity["label_share"][label]
        lines.append(f"  {label}: {count} ({share:.4f})")
    lines.append("OWASP category counts:")
    for cat, count in sanity["owasp_counts"].items():
        share = sanity["owasp_share"][cat]
        lines.append(f"  {cat}: {count} ({share:.4f})")
    lines.append("Chain distribution:")
    for chain, count in sanity["chain_counts"].items():
        lines.append(f"  {chain}: {count}")
    lines.append("Entity type distribution:")
    for etype, count in sanity["entity_counts"].items():
        lines.append(f"  {etype}: {count}")
    lines.append("Loss_usd stats:")
    for key, value in sanity["loss_stats"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("=== Простая ML-проверка (TF-IDF char 3-5 по risk_type + LogisticRegression, 80/20) ===")
    lines.append(f"Macro F1: {baseline_metrics['f1_macro']:.4f}")
    lines.append(f"ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
    lines.append("Classification report:")
    lines.append(baseline_report)
    lines.append("")
    lines.append("=== Stratified 5-fold CV (binary macro F1 / ROC-AUC) ===")
    lines.append(
        f"F1 macro: mean={cv_metrics['f1_macro_mean']:.4f}, std={cv_metrics['f1_macro_std']:.4f}"
    )
    lines.append(
        f"ROC-AUC: mean={cv_metrics['roc_auc_mean']:.4f}, std={cv_metrics['roc_auc_std']:.4f}"
    )
    return "\n".join(lines)


def evaluate_token_dataset(
    dataset_path: Path | None = None,
    report_path: Path | None = None,
) -> str:
    df = _load_token_dataset(dataset_path)
    sanity = _compute_sanity_stats(df)
    baseline_metrics, baseline_report = _train_test_baseline(df)
    cv_metrics = _kfold_cv(df)
    report_text = _format_report(
        sanity,
        baseline_metrics,
        baseline_report,
        cv_metrics,
    )
    output_path = report_path or (REPORTS_DIR / "token_risk_report.txt")
    output_path.write_text(report_text, encoding="utf-8")
    return str(output_path)


def main() -> None:
    evaluate_token_dataset()


if __name__ == "__main__":
    main()



