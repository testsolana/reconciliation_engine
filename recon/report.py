"""Формування вихідних артефактів: Excel, CSV, JSON, Markdown."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

SEV_FILL = {"HIGH": "F8C9C4", "MEDIUM": "FBE3B0", "LOW": "DCE9F7", "INFO": "E8E8E8"}


def _style(ws, sev_col: str | None = None, wrap_col: str | None = None):
    hdr = PatternFill("solid", fgColor="1F3A5F")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = hdr
    ws.freeze_panes = "A2"
    heads = {c.value: c.column for c in ws[1]}
    for name, col in heads.items():
        width = max(len(str(name)), *(len(str(ws.cell(r, col).value or "")) for r in range(2, min(ws.max_row, 80) + 1))) if ws.max_row > 1 else len(str(name))
        ws.column_dimensions[get_column_letter(col)].width = min(max(10, width + 2), 90 if name == wrap_col else 28)
    if wrap_col in heads:
        for r in range(2, ws.max_row + 1):
            ws.cell(r, heads[wrap_col]).alignment = Alignment(wrap_text=True, vertical="top")
    if sev_col in heads:
        for r in range(2, ws.max_row + 1):
            v = ws.cell(r, heads[sev_col]).value
            if v in SEV_FILL:
                ws.cell(r, heads[sev_col]).fill = PatternFill("solid", fgColor=SEV_FILL[v])
    ws.auto_filter.ref = ws.dimensions


def write_outputs(out: Path, exceptions, outliers, bridge, m, mm, clean_log, stats):
    out.mkdir(parents=True, exist_ok=True)
    exceptions.to_csv(out / "exceptions.csv", index=False)

    by_code = (exceptions.groupby(["severity", "category", "code"])
               .agg(count=("code", "size"), exposure_usd=("exposure_usd", "sum"), net_impact_usd=("impact_usd", "sum"))
               .reset_index())
    by_code["_r"] = by_code.severity.map({"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3})
    by_code = by_code.sort_values(["_r", "exposure_usd"], ascending=[True, False]).drop(columns="_r").round(2)

    matched = m[m.matched].copy()
    matched["gross_diff_usd"] = (matched.gross_amount_usd - matched.internal_gross_usd).round(2)
    matched["fx_rel_diff_pct"] = ((matched.provider_fx_rate_to_usd / matched.internal_fx_rate_to_usd - 1) * 100).round(3)
    matched["fee_rate_pct"] = (matched.provider_fee_usd / matched.gross_amount_usd.abs() * 100).round(3)
    matched["settlement_lag_h"] = ((matched.settled_at_utc - matched.created_at_utc).dt.total_seconds() / 3600).round(1)
    flagged = set(exceptions[exceptions.severity.isin(["HIGH", "MEDIUM", "LOW"])].transaction_id.dropna())
    matched["flagged"] = matched.transaction_id.isin(flagged)
    keep = ["transaction_id", "provider_record_id", "status", "transaction_type", "settlement_type", "original_currency",
            "original_amount", "internal_fx_rate_to_usd", "provider_fx_rate_to_usd", "fx_rel_diff_pct",
            "internal_gross_usd", "gross_amount_usd", "gross_diff_usd", "provider_fee_usd", "fee_rate_pct",
            "net_amount_usd", "settlement_lag_h", "flagged"]

    with pd.ExcelWriter(out / "reconciliation_report.xlsx", engine="openpyxl") as xw:
        pd.DataFrame(list(stats.items()), columns=["metric", "value"]).to_excel(xw, sheet_name="Summary", index=False)
        by_code.to_excel(xw, sheet_name="Findings_by_type", index=False)
        exceptions.to_excel(xw, sheet_name="Exceptions", index=False)
        bridge.to_excel(xw, sheet_name="Recon_bridge", index=False)
        outliers.sort_values("robust_z", key=abs, ascending=False).to_excel(xw, sheet_name="Outlier_review", index=False)
        matched[keep].to_excel(xw, sheet_name="Matched_pairs", index=False)
        clean_log.to_excel(xw, sheet_name="Cleaning_log", index=False)
        wb = xw.book
        _style(wb["Summary"]); _style(wb["Findings_by_type"], "severity")
        _style(wb["Exceptions"], "severity", "detail"); _style(wb["Recon_bridge"])
        _style(wb["Outlier_review"]); _style(wb["Matched_pairs"]); _style(wb["Cleaning_log"])

    summary = {"stats": stats, "findings_by_type": by_code.to_dict("records"),
               "bridge": bridge.to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return by_code
