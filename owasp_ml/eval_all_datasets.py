from __future__ import annotations

from pathlib import Path

from .eval import (
    evaluate_code_dataset,
    evaluate_token_dataset,
    evaluate_tx_dataset,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "owasp_ml/reports"


def evaluate_all_datasets() -> str:
    """
    Запускает оценку всех датасетов (code / token / tx)
    и формирует единый текстовый отчёт.
    Возвращает путь к сводному отчёту.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    code_report_path = Path(evaluate_code_dataset())
    token_report_path = Path(evaluate_token_dataset())
    tx_report_path = Path(evaluate_tx_dataset())

    parts: list[str] = []
    for report_path in (code_report_path, token_report_path, tx_report_path):
        if report_path.exists():
            parts.append(report_path.read_text(encoding="utf-8"))

    combined_report = "\n\n" + ("=" * 80) + "\n\n"
    combined_report = combined_report.join(parts) if parts else ""

    combined_path = REPORTS_DIR / "owasp_owasp_ml_full_report.txt"
    combined_path.write_text(combined_report, encoding="utf-8")
    return str(combined_path)


def main() -> None:
    evaluate_all_datasets()


if __name__ == "__main__":
    main()


