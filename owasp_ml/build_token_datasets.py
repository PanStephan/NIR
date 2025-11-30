from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "owasp_ml/prep"


def _clean_loss_usd(value: str | float | int | None) -> float | np.nan:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return np.nan
    cleaned = (
        text.replace("$", "")
        .replace(",", "")
        .replace(" ", "")
    )
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _normalize_chain(chain: str | None) -> str | None:
    if chain is None:
        return None
    text = str(chain).strip().upper()
    if not text:
        return None
    return text


def _merge_semicolon(values: Iterable[str | None]) -> str | None:
    uniques = []
    for v in values:
        if v is None:
            continue
        text = str(v).strip()
        if not text:
            continue
        if text not in uniques:
            uniques.append(text)
    if not uniques:
        return None
    return ";".join(uniques)


def _choose_entity_type(values: Iterable[str | None]) -> str | None:
    normalized = []
    for v in values:
        if v is None:
            continue
        text = str(v).strip().lower()
        if not text:
            continue
        normalized.append(text)
    if not normalized:
        return None
    if "token" in normalized:
        return "token"
    if "contract" in normalized:
        return "contract"
    if "wallet" in normalized:
        return "wallet"
    return normalized[0]


def _normalize_text_values(values: Iterable[str | None]) -> str:
    parts: list[str] = []
    for v in values:
        if v is None:
            continue
        text = str(v).strip().lower()
        if not text:
            continue
        parts.append(text)
    return " ".join(parts)


def _map_token_row_to_owasp_category(
    entity_type: str | None,
    risk_label: int | float | None,
    risk_type: str | None,
    root_cause: str | None,
) -> str | None:
    if risk_label in (0, 0.0):
        return "none"
    text = _normalize_text_values([risk_type, root_cause])
    if not text:
        return None
    if (
        "rug" in text
        or "rugpull" in text
        or "exit scam" in text
        or "exit-scam" in text
        or "pull" in text
        or "liquidity" in text
    ):
        return "onchain_rugpull"
    if "honeypot" in text:
        return "honeypot"
    if "phish" in text:
        return "phishing"
    if "ponzi" in text:
        return "ponzi_scheme"
    if "scam" in text or "fraud" in text:
        return "generic_scam"
    if (
        "exploit" in text
        or "reentrancy" in text
        or "overflow" in text
        or "underflow" in text
        or "bug" in text
        or "vulnerability" in text
    ):
        return "smart_contract_bug"
    if entity_type == "wallet":
        return "address_level_risk"
    return "other_onchain_risk"


def _map_tx_row_to_owasp_category(
    label: int | float | None,
    from_category: str | None,
    to_category: str | None,
) -> str | None:
    if label in (0, 0.0):
        return "none"
    text = _normalize_text_values([from_category, to_category])
    if not text:
        return "other_onchain_risk"
    if "phish" in text:
        return "phishing"
    if "scam" in text:
        return "generic_scam"
    if "mixer" in text or "tumbl" in text or "privacy" in text:
        return "money_laundering"
    if "dex" in text or "exchange" in text or "swap" in text:
        return "dex_abuse"
    return "other_onchain_risk"


def _load_rugpull_sources() -> pd.DataFrame:
    paths = [
        DATASETS_DIR / "rugpull_full_dataset_new.csv",
        DATASETS_DIR / "rugpull_dataset.csv",
    ]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df = df.rename(
            columns={
                "address": "address",
                "Chain": "chain",
                "Losses": "loss_usd_raw",
                "Type": "risk_type",
                "Root Causes": "root_cause",
                "Sources": "detail_source",
            }
        )
        df["address"] = df["address"].astype(str).str.lower()
        df["chain"] = df["chain"].map(_normalize_chain)
        df["loss_usd"] = df["loss_usd_raw"].map(_clean_loss_usd)
        df["entity_type"] = "token"
        df["risk_label"] = 1
        df["source"] = "rugpull"
        frames.append(
            df[
                [
                    "address",
                    "chain",
                    "entity_type",
                    "risk_label",
                    "risk_type",
                    "source",
                    "loss_usd",
                    "root_cause",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "address",
                "chain",
                "entity_type",
                "risk_label",
                "risk_type",
                "source",
                "loss_usd",
                "root_cause",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _load_malicious_smart_contracts() -> pd.DataFrame:
    path = DATASETS_DIR / "malicious_smart_contracts.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "address",
                "chain",
                "entity_type",
                "risk_label",
                "risk_type",
                "source",
                "loss_usd",
                "root_cause",
            ]
        )
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"contract_address": "address"})
    df["address"] = df["address"].astype(str).str.lower()
    df["chain"] = "ETH"
    df["entity_type"] = "contract"
    df["risk_label"] = 1
    df["source"] = "malicious_smart_contracts"
    raw_type = df["contract_creator_etherscan_label"].fillna("")
    raw_tag = df.get("contract_tag", pd.Series([""] * len(df))).fillna("")
    raw_source = df.get("source", pd.Series([""] * len(df))).fillna("")
    df["risk_type"] = np.where(
        raw_type.str.strip() != "",
        raw_type,
        np.where(
            raw_tag.str.strip() != "",
            raw_tag,
            raw_source,
        ),
    )
    df["risk_type"] = df["risk_type"].replace("", np.nan)
    df["loss_usd"] = np.nan
    df["root_cause"] = np.nan
    return df[
        [
            "address",
            "chain",
            "entity_type",
            "risk_label",
            "risk_type",
            "source",
            "loss_usd",
            "root_cause",
        ]
    ]


def _load_etherscan_malicious_like(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "address",
                "chain",
                "entity_type",
                "risk_label",
                "risk_type",
                "source",
                "loss_usd",
                "root_cause",
            ]
        )
    df = pd.read_csv(path)
    if "address" not in df.columns and "banned_address" in df.columns:
        df = df.rename(columns={"banned_address": "address"})
    df["address"] = df["address"].astype(str).str.lower()
    df["chain"] = "ETH"
    is_contract = df.get("is_contract")
    if is_contract is not None:
        df["entity_type"] = np.where(is_contract.astype(str) == "True", "contract", "wallet")
    else:
        df["entity_type"] = "wallet"
    df["risk_label"] = 1
    df["risk_type"] = "phishing"
    df["source"] = source_name
    df["loss_usd"] = np.nan
    df["root_cause"] = np.nan
    return df[
        [
            "address",
            "chain",
            "entity_type",
            "risk_label",
            "risk_type",
            "source",
            "loss_usd",
            "root_cause",
        ]
    ]


def _load_eth_legit_addresses() -> pd.DataFrame:
    path = DATASETS_DIR / "eth_addresses.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "address",
                "chain",
                "entity_type",
                "risk_label",
                "risk_type",
                "source",
                "loss_usd",
                "root_cause",
            ]
        )
    df = pd.read_csv(path)
    df = df.rename(columns={"Address": "address"})
    df["address"] = df["address"].astype(str).str.lower()
    df["chain"] = "ETH"
    account_type = df.get("Account Type", pd.Series([""] * len(df))).fillna("")
    contract_type = df.get("Contract Type", pd.Series([""] * len(df))).fillna("")
    account_type_norm = account_type.str.strip().str.lower()
    contract_type_norm = contract_type.str.strip().str.lower()
    entity_type = []
    for acc, ct in zip(account_type_norm, contract_type_norm):
        if acc == "smart contract":
            if ct == "token":
                entity_type.append("token")
            else:
                entity_type.append("contract")
        else:
            entity_type.append("wallet")
    df["entity_type"] = entity_type
    labels = df.get("Label", pd.Series([""] * len(df))).fillna("")
    label_norm = labels.str.strip().str.lower()
    df["risk_label"] = np.where(label_norm == "legit", 0, 1)
    df["risk_type"] = np.where(
        label_norm == "legit",
        "legit",
        labels,
    )
    df["source"] = "eth_addresses"
    df["loss_usd"] = np.nan
    df["root_cause"] = np.nan
    return df[
        [
            "address",
            "chain",
            "entity_type",
            "risk_label",
            "risk_type",
            "source",
            "loss_usd",
            "root_cause",
        ]
    ]


def _load_tools_ux_dataset() -> pd.DataFrame:
    path = DATASETS_DIR / "tools_UX.csv"
    if not path.exists():
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
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Smart Contract": "address",
            "Function Name": "function_name",
        }
    )
    tool_columns = [
        "slither",
        "solhint",
        "mythril",
        "conkas",
        "smartcheck",
        "Actual",
    ]
    for col in tool_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = (
            df[col]
            .replace("N/A", np.nan)
            .fillna(0)
            .astype(int)
        )
    df = df.rename(
        columns={
            "slither": "slither_pred",
            "solhint": "solhint_pred",
            "mythril": "mythril_pred",
            "conkas": "conkas_pred",
            "smartcheck": "smartcheck_pred",
            "Actual": "actual_label",
        }
    )
    df["address"] = df["address"].astype(str).str.strip().str.lower()
    df["function_name"] = df["function_name"].astype(str).str.strip()
    df = df[df["function_name"] != ""]
    df = df.dropna(subset=["actual_label"])
    df = df.drop_duplicates()
    return df[
        [
            "address",
            "function_name",
            "slither_pred",
            "solhint_pred",
            "mythril_pred",
            "conkas_pred",
            "smartcheck_pred",
            "actual_label",
        ]
    ]


def build_token_risk_dataset() -> pd.DataFrame:
    frames = [
        _load_rugpull_sources(),
        _load_malicious_smart_contracts(),
        _load_etherscan_malicious_like(
            DATASETS_DIR / "etherscan_malicious_labels.csv",
            "etherscan_malicious",
        ),
        _load_etherscan_malicious_like(
            DATASETS_DIR / "phishing_scams.csv",
            "phishing_scams",
        ),
        _load_eth_legit_addresses(),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "address",
                "chain",
                "entity_type",
                "risk_label",
                "risk_type",
                "source",
                "loss_usd",
                "root_cause",
                "owasp_category",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    grouped = df.groupby("address", dropna=False)
    agg = grouped.agg(
        chain=("chain", _merge_semicolon),
        entity_type=("entity_type", _choose_entity_type),
        risk_label=("risk_label", "max"),
        risk_type=("risk_type", _merge_semicolon),
        source=("source", _merge_semicolon),
        loss_usd=("loss_usd", "max"),
        root_cause=("root_cause", _merge_semicolon),
    ).reset_index()
    agg["owasp_category"] = [
        _map_token_row_to_owasp_category(
            entity_type=row["entity_type"],
            risk_label=row["risk_label"],
            risk_type=row["risk_type"],
            root_cause=row["root_cause"],
        )
        for _, row in agg.iterrows()
    ]
    return agg[
        [
            "address",
            "chain",
            "entity_type",
            "risk_label",
            "risk_type",
            "source",
            "loss_usd",
            "root_cause",
            "owasp_category",
        ]
    ]


def build_tx_scam_dataset() -> pd.DataFrame:
    path = DATASETS_DIR / "Dataset.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "tx_hash",
                "from_address",
                "to_address",
                "value",
                "gas",
                "gas_price",
                "block_timestamp",
                "block_number",
                "label",
                "from_category",
                "to_category",
                "owasp_category",
            ]
        )
    df = pd.read_csv(path)
    df["hash"] = df["hash"].astype(str).str.strip().str.lower()
    df = df[df["hash"] != ""]
    df = df.drop_duplicates(subset=["hash"])
    df["label"] = (
        df["from_scam"].fillna(0).astype(int)
        | df["to_scam"].fillna(0).astype(int)
    )
    result = pd.DataFrame()
    result["tx_hash"] = df["hash"]
    result["from_address"] = df["from_address"]
    result["to_address"] = df["to_address"]
    result["value"] = df["value"]
    result["gas"] = df["gas"]
    result["gas_price"] = df["gas_price"]
    result["block_timestamp"] = df["block_timestamp"]
    result["block_number"] = df["block_number"]
    result["from_address"] = result["from_address"].astype(str).str.strip().str.lower()
    result["to_address"] = result["to_address"].astype(str).str.strip().str.lower()
    result["label"] = df["label"]
    result["from_category"] = df["from_category"]
    result["to_category"] = df["to_category"]
    result["owasp_category"] = [
        _map_tx_row_to_owasp_category(
            label=row["label"],
            from_category=row["from_category"],
            to_category=row["to_category"],
        )
        for _, row in result.iterrows()
    ]
    return result


def build_code_risk_dataset() -> pd.DataFrame:
    path = DATASETS_DIR / "SC_4label.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "filename",
                "code",
                "label",
                "label_encoded",
                "code_length_chars",
                "code_length_lines",
                "owasp_category",
            ]
        )
    df = pd.read_csv(path)
    if "filename" not in df.columns and df.columns[1] == "filename":
        df = df.iloc[:, 1:]
    df["filename"] = df["filename"].astype(str)
    df["code"] = df["code"].astype(str)
    df = df.dropna(subset=["code", "label"])
    df["filename"] = df["filename"].str.strip()
    df["code"] = df["code"].str.strip()
    df = df[df["code"] != ""]
    df = df.drop_duplicates(subset=["filename", "code", "label"])
    df["code_length_chars"] = df["code"].str.len()
    df["code_length_lines"] = df["code"].str.count("\n") + 1
    df["owasp_category"] = df["label"].astype(str)
    return df[
        [
            "filename",
            "code",
            "label",
            "label_encoded",
            "code_length_chars",
            "code_length_lines",
            "owasp_category",
        ]
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    token_df = build_token_risk_dataset()
    tx_df = build_tx_scam_dataset()
    code_df = build_code_risk_dataset()
    tools_ux_df = _load_tools_ux_dataset()
    token_df.to_csv(OUTPUT_DIR / "token_risk_dataset.csv", index=False)
    tx_df.to_csv(OUTPUT_DIR / "tx_scam_dataset.csv", index=False)
    code_df.to_csv(OUTPUT_DIR / "code_risk_dataset.csv", index=False)
    tools_ux_df.to_csv(OUTPUT_DIR / "tools_ux_dataset.csv", index=False)


if __name__ == "__main__":
    main()


