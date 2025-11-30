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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


PKG_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = PKG_DIR / "prep"
REPORTS_DIR = PKG_DIR


def _load_code_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or (DATASETS_DIR / "code_risk_dataset.csv")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["code", "label"])
    df["code"] = df["code"].astype(str)
    return df


def _compute_sanity_stats(df: pd.DataFrame) -> Dict[str, object]:
    total_samples = int(len(df))
    label_counts = df["label"].value_counts().sort_index()
    label_share = (label_counts / total_samples).to_dict()
    length_stats = df["code_length_chars"].describe().to_dict()
    per_label_examples: Dict[str, List[str]] = {}
    for label_value, group in df.groupby("label"):
        examples = group.sort_values("code_length_chars", ascending=False)["filename"].head(3).tolist()
        per_label_examples[str(label_value)] = examples
    return {
        "total_samples": total_samples,
        "label_counts": {str(k): int(v) for k, v in label_counts.to_dict().items()},
        "label_share": label_share,
        "length_stats": length_stats,
        "per_label_examples": per_label_examples,
    }


def _train_test_baseline(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], str]:
    X = df["code"].tolist()
    y = df["label"].astype(str).tolist()
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
        min_df=5,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = LogisticRegression(
        max_iter=1000,
    )
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    labels_sorted = sorted(set(y_train) | set(y_test))
    y_test_encoded = np.array([labels_sorted.index(v) for v in y_test])
    proba = classifier.predict_proba(X_test_vec)
    roc_auc_macro = roc_auc_score(y_test_encoded, proba, multi_class="ovr", average="macro")
    metrics = {
        "f1_macro": float(f1_macro),
        "roc_auc_macro": float(roc_auc_macro),
    }
    return metrics, report_text


def _kfold_cv(df: pd.DataFrame) -> Dict[str, float]:
    X = df["code"].tolist()
    y = df["label"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
    )
    X_vec = vectorizer.fit_transform(X)
    labels_sorted = sorted(set(y))
    y_encoded = np.array([labels_sorted.index(v) for v in y])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores: List[float] = []
    auc_scores: List[float] = []
    for train_idx, test_idx in skf.split(X_vec, y_encoded):
        X_train = X_vec[train_idx]
        X_test = X_vec[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))
        proba = clf.predict_proba(X_test)
        auc_scores.append(
            roc_auc_score(
                y_test,
                proba,
                multi_class="ovr",
                average="macro",
            )
        )
    return {
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "roc_auc_macro_mean": float(np.mean(auc_scores)),
        "roc_auc_macro_std": float(np.std(auc_scores)),
    }


def _per_length_bin_f1(df: pd.DataFrame) -> Dict[str, float]:
    quantiles = df["code_length_chars"].quantile([0.25, 0.5, 0.75]).to_dict()
    bins = [
        float(df["code_length_chars"].min()),
        float(quantiles[0.25]),
        float(quantiles[0.5]),
        float(quantiles[0.75]),
        float(df["code_length_chars"].max()),
    ]
    labels_bin = ["q0_q25", "q25_q50", "q50_q75", "q75_q100"]
    df = df.copy()
    df["length_bin"] = pd.cut(
        df["code_length_chars"],
        bins=bins,
        labels=labels_bin,
        include_lowest=True,
        duplicates="drop",
    )
    results: Dict[str, float] = {}
    for bin_label, group in df.groupby("length_bin", observed=False):
        if group.empty:
            continue
        metrics, _ = _train_test_baseline(group)
        results[str(bin_label)] = metrics["f1_macro"]
    return results


def _format_report(
    sanity: Dict[str, object],
    baseline_metrics: Dict[str, float],
    baseline_report: str,
    cv_metrics: Dict[str, float],
    length_bin_f1: Dict[str, float],
    tools_metrics: Dict[str, Dict[str, float]],
) -> str:
    lines: List[str] = []
    lines.append("=== Sanity-проверка code_risk_dataset ===")
    lines.append(f"Total samples: {sanity['total_samples']}")
    lines.append("Label counts:")
    for label, count in sanity["label_counts"].items():
        share = sanity["label_share"][label]
        lines.append(f"  {label}: {count} ({share:.4f})")
    lines.append("Code length (chars) stats:")
    for key, value in sanity["length_stats"].items():
        lines.append(f"  {key}: {value}")
    lines.append("Примеры файлов по классам:")
    for label, examples in sanity["per_label_examples"].items():
        joined = ", ".join(examples)
        lines.append(f"  label={label}: {joined}")
    lines.append("")
    lines.append("=== Простая ML-проверка (TF-IDF char 3-5 + LogisticRegression, 80/20) ===")
    lines.append(f"Macro F1: {baseline_metrics['f1_macro']:.4f}")
    lines.append(f"Macro ROC-AUC (OvR): {baseline_metrics['roc_auc_macro']:.4f}")
    lines.append("Classification report:")
    lines.append(baseline_report)
    lines.append("")
    lines.append("=== Stratified 5-fold CV (macro F1 / ROC-AUC) ===")
    lines.append(
        f"F1 macro: mean={cv_metrics['f1_macro_mean']:.4f}, std={cv_metrics['f1_macro_std']:.4f}"
    )
    lines.append(
        f"ROC-AUC macro: mean={cv_metrics['roc_auc_macro_mean']:.4f}, std={cv_metrics['roc_auc_macro_std']:.4f}"
    )
    lines.append("")
    lines.append("=== Качество по длине контракта (F1 macro по квантильным бинам) ===")
    for bin_label, score in length_bin_f1.items():
        lines.append(f"  {bin_label}: F1 macro = {score:.4f}")
    lines.append("")
    lines.append("=== Оценка open-source инструментов по датасету tools_ux_dataset ===")
    if not tools_metrics:
        lines.append("Нет данных для оценки инструментов.")
    else:
        for tool_col, m in tools_metrics.items():
            name = tool_col.replace("_pred", "")
            lines.append(
                f"  {name}: precision={m['precision']:.4f}, recall={m['recall']:.4f}, f1={m['f1']:.4f}"
            )
    return "\n".join(lines)


def _load_tools_ux_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or (DATASETS_DIR / "tools_ux_dataset.csv")
    if not dataset_path.exists():
        return pd.DataFrame(
            columns=[
                "address",
                "function_name",
                "slither_pred",
                "solhint_pred",
                "mythril_pred",
                "conkas_pred",
                "smartcheck_pred",
                "actual_label",
            ]
        )
    df = pd.read_csv(dataset_path)
    return df


def _compute_tools_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}
    y_true = df["actual_label"].astype(int).to_numpy()
    tools = [
        "slither_pred",
        "solhint_pred",
        "mythril_pred",
        "conkas_pred",
        "smartcheck_pred",
    ]
    results: Dict[str, Dict[str, float]] = {}
    for col in tools:
        if col not in df.columns:
            continue
        y_pred = df[col].astype(int).to_numpy()
        if y_pred.max() == 0 and y_true.max() == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        results[col] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return results


def evaluate_code_dataset(
    dataset_path: Path | None = None,
    report_path: Path | None = None,
) -> str:
    df = _load_code_dataset(dataset_path)
    sanity = _compute_sanity_stats(df)
    baseline_metrics, baseline_report = _train_test_baseline(df)
    cv_metrics = _kfold_cv(df)
    length_bin_f1 = _per_length_bin_f1(df)
    tools_df = _load_tools_ux_dataset()
    tools_metrics = _compute_tools_metrics(tools_df)
    report_text = _format_report(
        sanity,
        baseline_metrics,
        baseline_report,
        cv_metrics,
        length_bin_f1,
        tools_metrics,
    )
    output_path = report_path or (REPORTS_DIR / "code_risk_report.txt")
    output_path.write_text(report_text, encoding="utf-8")
    return str(output_path)


def main() -> None:
    evaluate_code_dataset()


if __name__ == "__main__":
    main()


