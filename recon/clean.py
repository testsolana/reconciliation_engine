"""Завантаження та нормалізація даних. Кожна зміна фіксується в cleaning log."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

INTERNAL_REQUIRED = [
    "transaction_id", "user_id", "created_at_utc", "transaction_type", "status",
    "original_currency", "original_amount", "internal_fx_rate_to_usd", "internal_gross_usd",
    "merchant_country", "merchant_id", "merchant_category", "channel",
]
PROVIDER_REQUIRED = [
    "provider_record_id", "client_reference", "settled_at_utc", "settlement_type", "provider_status",
    "settlement_currency", "gross_amount_usd", "provider_fee_usd", "net_amount_usd",
    "provider_fx_rate_to_usd", "merchant_country", "batch_id",
]

_ID_JUNK = re.compile(r"[\s​ ]+")


def _to_number(s: pd.Series) -> pd.Series:
    """'1,234.50', '$12', '(5.00)' -> float. Нечислове -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.strip()
    neg = t.str.match(r"^\(.*\)$")
    t = t.str.replace(r"[()$€£,\s]", "", regex=True)
    out = pd.to_numeric(t, errors="coerce")
    return out.where(~neg, -out)


def _clean_frame(df: pd.DataFrame, name: str, required: list[str], ids: list[str],
                 enums: list[str], numbers: list[str], times: list[str], log: list[dict]) -> pd.DataFrame:
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: відсутні обов'язкові колонки {missing_cols}")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def note(col, action, n):
        if n:
            log.append({"dataset": name, "column": col, "action": action, "rows_affected": int(n)})

    for c in ids:
        raw = df[c].astype("string")
        new = raw.str.replace(_ID_JUNK, "", regex=True).str.upper()
        note(c, "trim/upper (пробіли, zero-width, регістр)", (raw != new).sum())
        df[c] = new
    for c in enums:
        raw = df[c].astype("string")
        new = raw.str.strip().str.upper()
        note(c, "trim/upper", (raw != new).sum())
        df[c] = new
    for c in numbers:
        before = df[c].isna().sum()
        new = _to_number(df[c])
        note(c, "coerce to float (поламані значення -> NaN)", new.isna().sum() - before)
        df[c] = new
    for c in times:
        before = df[c].isna().sum()
        new = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)
        note(c, "parse datetime (UTC, naive)", new.isna().sum() - before)
        df[c] = new
    for c in required:
        note(c, "NULL у обов'язковому полі", df[c].isna().sum())
    return df


def load_and_clean(path: str, internal_sheet: str, provider_sheet: str):
    log: list[dict] = []
    raw_i = pd.read_excel(path, sheet_name=internal_sheet, dtype=object)
    raw_p = pd.read_excel(path, sheet_name=provider_sheet, dtype=object)

    i = _clean_frame(
        raw_i, "internal", INTERNAL_REQUIRED,
        ids=["transaction_id", "user_id", "merchant_id"],
        enums=["transaction_type", "status", "original_currency", "card_country", "merchant_country", "channel"],
        numbers=["original_amount", "internal_fx_rate_to_usd", "internal_gross_usd"],
        times=["created_at_utc"], log=log)
    i["merchant_category"] = i["merchant_category"].astype(str).str.strip()

    p = _clean_frame(
        raw_p, "provider", PROVIDER_REQUIRED,
        ids=["provider_record_id", "client_reference", "batch_id"],
        enums=["settlement_type", "provider_status", "settlement_currency", "merchant_country"],
        numbers=["gross_amount_usd", "provider_fee_usd", "net_amount_usd", "provider_fx_rate_to_usd"],
        times=["settled_at_utc"], log=log)

    # Повні дублікати рядків (не видаляємо — позначаємо; рішення про видалення за checks)
    i["row_dup_full"] = i.duplicated(keep="first")
    p["row_dup_full"] = p.duplicated(keep="first")
    log.append({"dataset": "meta", "column": "-", "action": f"internal rows={len(i)}, provider rows={len(p)}",
                "rows_affected": 0})
    return i, p, pd.DataFrame(log)
