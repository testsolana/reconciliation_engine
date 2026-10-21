"""Matching + all anomaly detectors.

impact_usd convention: provider - internal (positive = provider settled more than our ledger booked).
For MISSING_SETTLEMENT impact = -internal gross (money we expected but did not receive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# code -> (category, base severity, "what this means for the business")
CATALOG = {
    "DUPLICATE_SETTLEMENT": ("Reconciliation", "HIGH", "One payment settled twice: risk of double debit/payout."),
    "ORPHAN_SETTLEMENT": ("Reconciliation", "HIGH", "Provider settled a transaction that is not in the ledger: fraud or bookkeeping-failure risk."),
    "DECLINED_SETTLED": ("Reconciliation", "HIGH", "A declined transaction was settled: money moved without a valid transaction."),
    "MISSING_SETTLEMENT": ("Reconciliation", "HIGH", "A COMPLETED transaction was not settled by the provider: funds not received."),
    "SIGN_OR_TYPE_MISMATCH": ("Reconciliation", "HIGH", "Type/sign mismatch (purchase<->refund): balance effect up to 2x the amount."),
    "AMOUNT_MISMATCH": ("Reconciliation", "MEDIUM", "Provider gross differs from ledger beyond the FX-noise tolerance."),
    "FX_MISMATCH": ("FX", "MEDIUM", "Provider FX rate deviates from the ledger and from the currency's normal range."),
    "FX_INTERNAL_OUT_OF_BAND": ("FX", "MEDIUM", "Internal FX rate is outside the currency's normal range."),
    "FEE_ANOMALY": ("Fee", "MEDIUM", "Fee matches no known fee tier or minimum fee."),
    "NET_MISMATCH": ("Fee", "MEDIUM", "net != gross -/+ fee: arithmetic error on the provider side."),
    "LATE_SETTLEMENT": ("Timing", "LOW", "Settlement far later than normal: cash-flow impact and risk of a lost record."),
    "SETTLED_BEFORE_CREATED": ("Timing", "MEDIUM", "Settlement precedes transaction creation: timestamp or identifier problem."),
    "POSSIBLE_DUPLICATE_INTERNAL": ("Duplicates", "MEDIUM", "Near-identical operations by one customer within minutes: possible double charge."),
    "MERCHANT_COUNTRY_MISMATCH": ("Reconciliation", "LOW", "Merchant country differs between the two systems."),
    "INTERNAL_CALC_ERROR": ("Data quality", "MEDIUM", "original_amount × fx ≠ internal_gross_usd."),
    "INTERNAL_SIGN_ERROR": ("Data quality", "MEDIUM", "Amount sign does not match the transaction type (purchase>0, refund<0)."),
    "PROVIDER_CALC_ERROR": ("Data quality", "MEDIUM", "Provider gross != original_amount x provider_fx."),
    "BATCH_DATE_MISMATCH": ("Data quality", "LOW", "batch_id does not match the settlement date."),
    "DUPLICATE_KEY": ("Data quality", "HIGH", "Duplicated key (transaction_id / provider_record_id)."),
    "NULL_KEY": ("Data quality", "HIGH", "Empty key: the row cannot be matched to the other system; it is reported as unmatched instead of being dropped."),
    "INVALID_VALUE": ("Data quality", "MEDIUM", "Null, zero or non-positive value in a field the checks depend on: the row cannot be compared reliably."),
    "HIGH_VALUE_OUTLIER": ("Outlier", "INFO", "Statistical outlier that reconciles cleanly: legitimate, but needs a business review."),
    "OUTLIER_WITH_DEFECT": ("Outlier", "HIGH", "Outlier that also has a reconciliation defect: investigate first."),
    "IN_FLIGHT": ("Reconciliation", "INFO", "COMPLETED without settlement but still inside the normal settlement window."),
}
SEV_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}


class Findings:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, code, txn=None, prov=None, impact=0.0, detail="", severity=None, exposure=None, **extra):
        cat, sev, meaning = CATALOG[code]
        self.rows.append({
            "severity": severity or sev, "category": cat, "code": code,
            "transaction_id": txn, "provider_record_id": prov,
            "impact_usd": round(float(impact), 2),
            "exposure_usd": round(abs(float(impact) if exposure is None else float(exposure)), 2),
            "detail": detail, "business_meaning": meaning, **extra})

    def frame(self) -> pd.DataFrame:
        cols = ["severity", "category", "code", "transaction_id", "provider_record_id",
                "impact_usd", "exposure_usd", "detail", "business_meaning"]
        df = pd.DataFrame(self.rows, columns=cols)
        if df.empty:
            return df
        df["_r"] = df.severity.map(SEV_RANK)
        return df.sort_values(["_r", "exposure_usd"], ascending=[True, False]).drop(columns="_r").reset_index(drop=True)


def robust_z(x: pd.Series) -> pd.Series:
    """Modified z-score (Iglewicz-Hoaglin): robust to the outliers themselves."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


def safe_div(a, b, eps: float = 1e-12):
    """a / b that returns NaN (never inf / ZeroDivisionError) when b is zero, null or ~0.
    Works for scalars and pandas Series; NaN is then handled explicitly by the callers."""
    if np.isscalar(b) or b is None:
        return float("nan") if (b is None or pd.isna(b) or abs(b) < eps) else a / b
    b = pd.Series(b)
    return a / b.where(b.abs() >= eps)


def _guard_inputs(i: pd.DataFrame, p: pd.DataFrame, F, ki: str, kp: str):
    """Make bad rows visible instead of letting them vanish silently.

    1. Empty keys get a unique placeholder (`__NULL_<side><row>`): they can never be merged with each
       other (pandas merges NaN == NaN) and never fall out of a groupby (which drops NaN groups).
       Each one is reported as NULL_KEY and then flows through matching as an unmatched/orphan row.
    2. Null / zero / non-positive values in fields the checks rely on are reported as INVALID_VALUE,
       because a NaN comparison is always False and the row would otherwise pass every check unnoticed.
    """
    i, p = i.copy(), p.copy()
    n_keys = 0
    for df, col, tag, who in ((i, ki, "I", "internal"), (p, kp, "P", "provider"),
                              (p, "provider_record_id", "R", "provider")):
        blank = (df[col].astype("string").str.strip() == "").fillna(False)
        for idx in df.index[df[col].isna() | blank]:
            ph = f"__NULL_{tag}{idx}"
            if who == "internal":
                txn, prov = ph, None
            elif col == kp:
                txn, prov = ph, df.at[idx, "provider_record_id"]
            else:
                txn, prov = df.at[idx, kp], ph
            F.add("NULL_KEY", txn=txn, prov=prov,
                  detail=f"{who} data row {idx + 1}: {col} is empty; placeholder {ph} is used so the row is "
                         f"reported as unmatched instead of being merged or dropped")
            df.loc[idx, col] = ph
            n_keys += 1

    spec = (
        (i, "internal", ["user_id", "transaction_type", "status", "original_currency", "merchant_id", "created_at_utc",
                         "original_amount", "internal_fx_rate_to_usd", "internal_gross_usd"],
         ["internal_fx_rate_to_usd"], ["internal_gross_usd"]),
        (p, "provider", ["settlement_type", "settled_at_utc", "batch_id", "gross_amount_usd", "provider_fee_usd",
                         "net_amount_usd", "provider_fx_rate_to_usd"],
         ["provider_fx_rate_to_usd"], ["gross_amount_usd"]),
    )
    n_bad = 0
    for df, who, required, positive, nonzero in spec:
        probs: dict = {}
        for c in required:
            for idx in df.index[df[c].isna()]:
                probs.setdefault(idx, []).append(f"{c} is null")
        for c in positive:
            for idx in df.index[df[c].notna() & (df[c] <= 0)]:
                probs.setdefault(idx, []).append(f"{c} = {df.at[idx, c]} (must be > 0)")
        for c in nonzero:
            for idx in df.index[df[c].notna() & (df[c] == 0)]:
                probs.setdefault(idx, []).append(f"{c} is zero")
        for idx in sorted(probs):
            F.add("INVALID_VALUE",
                  txn=df.at[idx, ki if who == "internal" else kp],
                  prov=df.at[idx, "provider_record_id"] if who == "provider" else None,
                  detail=f"{who} data row {idx + 1}: " + "; ".join(probs[idx]))
            n_bad += 1
    return i, p, {"rows_with_null_keys": n_keys, "rows_with_invalid_values": n_bad}


def _fee_fits(fee, gross, cfg) -> bool:
    g = abs(gross)
    for t in cfg["tiers"]:
        exp = max(cfg["min_fee_usd"], round(t * g, 2))
        if abs(fee - exp) <= max(cfg["rounding_usd"], cfg["tier_rel_tolerance"] * exp):
            return True
    return False


def build_match(i: pd.DataFrame, p: pd.DataFrame, cfg: dict):
    ki, kp = cfg["matching"]["key_internal"], cfg["matching"]["key_provider"]
    p = p.sort_values(["settled_at_utc", "provider_record_id"]).copy()
    p["_occ"] = p.groupby(kp).cumcount()
    primary = p[p._occ == 0]
    extras = p[p._occ > 0]
    m = i.merge(primary.drop(columns="_occ"), how="left", left_on=ki, right_on=kp,
                suffixes=("", "_p"), indicator=True)
    m["matched"] = m["_merge"] == "both"
    m = m.drop(columns="_merge")
    orphans = primary[~primary[kp].isin(i[ki])]
    return m, extras, orphans, p


def run_checks(i: pd.DataFrame, p: pd.DataFrame, cfg: dict):
    F = Findings()
    i, p = i.reset_index(drop=True), p.reset_index(drop=True)  # the code below relies on a unique index
    ki, kp = cfg["matching"]["key_internal"], cfg["matching"]["key_provider"]
    i, p, guard_stats = _guard_inputs(i, p, F, ki, kp)
    m, extras, orphans, p_sorted = build_match(i, p, cfg)
    # Row accounting: nothing may be lost or multiplied by matching
    n_primary = int((p_sorted._occ == 0).sum())
    if len(m) != len(i) or n_primary + len(extras) != len(p):
        raise RuntimeError(f"row accounting failed: internal {len(i)} -> merged {len(m)}; "
                           f"provider {len(p)} -> primary {n_primary} + duplicates {len(extras)}")
    A, FX, FEE, T, O = cfg["amount"], cfg["fx"], cfg["fee"], cfg["timing"], cfg["outliers"]

    # ---------- Data quality: internal ledger ----------
    for k in i.loc[i[ki].duplicated(keep=False), ki].unique():
        F.add("DUPLICATE_KEY", txn=k, detail=f"transaction_id {k} appears {int((i[ki] == k).sum())} times")
    for k in p.loc[p.provider_record_id.duplicated(keep=False), "provider_record_id"].unique():
        F.add("DUPLICATE_KEY", prov=k, detail=f"provider_record_id {k} is repeated")

    calc = (i.original_amount * i.internal_fx_rate_to_usd).round(2)
    bad = (calc - i.internal_gross_usd).abs() > 0.011
    for r in i[bad].itertuples():
        F.add("INTERNAL_CALC_ERROR", txn=r.transaction_id,
              impact=r.internal_gross_usd - calc[r.Index],
              detail=f"{r.original_amount}×{r.internal_fx_rate_to_usd}={calc[r.Index]} ≠ {r.internal_gross_usd}")
    sgn_bad = ((i.transaction_type == "PURCHASE") & (i.original_amount < 0)) | \
              ((i.transaction_type == "REFUND") & (i.original_amount > 0))
    for r in i[sgn_bad].itertuples():
        F.add("INTERNAL_SIGN_ERROR", txn=r.transaction_id, impact=r.internal_gross_usd,
              detail=f"{r.transaction_type} with amount {r.original_amount}")

    # FX band for internal rates (per currency, relative deviation from the median)
    med_rate = i.groupby("original_currency").internal_fx_rate_to_usd.median()
    i_dev = safe_div(i.internal_fx_rate_to_usd, i.original_currency.map(med_rate)) - 1
    for r in i[(i_dev.abs() > FX["band_min_rel"])].itertuples():
        F.add("FX_INTERNAL_OUT_OF_BAND", txn=r.transaction_id,
              detail=f"{r.original_currency} rate {r.internal_fx_rate_to_usd} vs median {med_rate[r.original_currency]:.4f} ({i_dev[r.Index]:+.2%})")

    # ---------- Data quality: provider ----------
    for r in p_sorted[p_sorted.settled_at_utc.dt.strftime("B%Y%m%d") != p_sorted.batch_id].itertuples():
        F.add("BATCH_DATE_MISMATCH", prov=r.provider_record_id, txn=r.client_reference,
              detail=f"settled {r.settled_at_utc:%Y-%m-%d}, batch {r.batch_id}")

    # ---------- Reconciliation: structural ----------
    for r in extras.itertuples():
        first = p_sorted[(p_sorted[kp] == r.client_reference) & (p_sorted._occ == 0)].iloc[0]
        same = (first.gross_amount_usd == r.gross_amount_usd and first.settled_at_utc == r.settled_at_utc)
        F.add("DUPLICATE_SETTLEMENT", txn=r.client_reference, prov=r.provider_record_id, impact=r.gross_amount_usd,
              detail=f"{r.provider_record_id} repeats {first.provider_record_id} "
                     f"({'identical amount and time' if same else 'different amount/time'}); extra gross {r.gross_amount_usd}")
    for r in orphans.itertuples():
        F.add("ORPHAN_SETTLEMENT", txn=r.client_reference, prov=r.provider_record_id, impact=r.gross_amount_usd,
              detail=f"client_reference {r.client_reference} is not in the ledger; gross {r.gross_amount_usd}, fee {r.provider_fee_usd}")

    dec = m[m.matched & (m.status == "DECLINED")]
    for r in dec.itertuples():
        F.add("DECLINED_SETTLED", txn=r.transaction_id, prov=r.provider_record_id, impact=r.gross_amount_usd,
              detail=f"internal status DECLINED, but settlement {r.provider_status} gross {r.gross_amount_usd}")

    data_end = p.settled_at_utc.max()
    unsettled = m[(~m.matched) & (m.status == "COMPLETED")]
    for r in unsettled.itertuples():
        age = (data_end - r.created_at_utc).total_seconds() / 3600
        if age > T["in_flight_grace_hours"]:
            F.add("MISSING_SETTLEMENT", txn=r.transaction_id, impact=-r.internal_gross_usd,
                  detail=f"COMPLETED {r.created_at_utc:%Y-%m-%d %H:%M}; absent from the provider report (through {data_end:%Y-%m-%d}), age {age:.0f}h")
        else:
            F.add("IN_FLIGHT", txn=r.transaction_id, impact=0,
                  detail=f"age {age:.0f}h < {T['in_flight_grace_hours']}h")

    # ---------- Reconciliation: amount comparison for matched COMPLETED pairs ----------
    mm = m[m.matched & (m.status == "COMPLETED")].copy()
    mm["gross_diff"] = mm.gross_amount_usd - mm.internal_gross_usd
    mm["fx_rel_diff"] = safe_div(mm.provider_fx_rate_to_usd, mm.internal_fx_rate_to_usd) - 1
    mm["tol"] = np.maximum(A["abs_tolerance_usd"], A["rel_tolerance"] * mm.internal_gross_usd.abs())
    mm["implied_gross"] = (mm.original_amount * mm.provider_fx_rate_to_usd).round(2)
    prov_dev = safe_div(mm.provider_fx_rate_to_usd, mm.original_currency.map(med_rate)) - 1
    mm["lag_h"] = (mm.settled_at_utc - mm.created_at_utc).dt.total_seconds() / 3600

    flagged_fx = set()
    for r in mm.itertuples():
        pdv = prov_dev[r.Index]
        if abs(r.fx_rel_diff) > FX["rel_tolerance_vs_internal"] and abs(pdv) > FX["band_min_rel"]:
            flagged_fx.add(r.transaction_id)
            F.add("FX_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id,
                  impact=r.gross_amount_usd - r.internal_gross_usd,
                  detail=f"{r.original_currency}: provider rate {r.provider_fx_rate_to_usd:.4f} vs internal "
                         f"{r.internal_fx_rate_to_usd:.4f} ({r.fx_rel_diff:+.2%}); provider {pdv:+.2%} from currency median, "
                         f"internal {i_dev[r.Index]:+.2%} -> provider is the deviating side")

    for r in mm.itertuples():
        if pd.isna(r.gross_amount_usd) or pd.isna(r.internal_gross_usd):
            continue  # reported as INVALID_VALUE; a NaN comparison would otherwise pass silently or fake a mismatch
        sign_flip = np.sign(r.gross_amount_usd) * np.sign(r.internal_gross_usd) < 0  # zero is not a sign flip
        type_diff = bool(pd.notna(r.settlement_type) and pd.notna(r.transaction_type)
                         and r.settlement_type != r.transaction_type)
        if sign_flip or type_diff:
            F.add("SIGN_OR_TYPE_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id,
                  impact=r.gross_amount_usd - r.internal_gross_usd,
                  detail=f"ledger {r.transaction_type} {r.internal_gross_usd} vs provider {r.settlement_type} {r.gross_amount_usd}"
                         f"{' (sign inverted)' if sign_flip else ''}; balance shift {r.gross_amount_usd - r.internal_gross_usd:+.2f}"
                         f"{'' if sign_flip else '; if REFUND is treated as an outgoing payment, risk up to 2x' + str(abs(r.internal_gross_usd))}",
                  exposure=abs(r.gross_amount_usd - r.internal_gross_usd) if sign_flip else 2 * abs(r.internal_gross_usd))
        elif abs(r.gross_diff) > r.tol:
            if r.transaction_id in flagged_fx and abs(r.implied_gross - r.gross_amount_usd) <= 0.02:
                continue  # fully explained by FX_MISMATCH (same impact) - do not double-report
            elif abs(r.implied_gross - r.gross_amount_usd) > 0.02:
                cause = (f"gross is not derivable from original x provider_fx (={r.implied_gross}) -> amount error, not FX")
            else:
                cause = "FX is normal"
            F.add("AMOUNT_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id, impact=r.gross_diff,
                  detail=f"ledger {r.internal_gross_usd} vs provider {r.gross_amount_usd} (Δ{r.gross_diff:+.2f}, tolerance ±{r.tol:.2f}); {cause}",
                  severity="HIGH" if abs(r.gross_diff) >= cfg["severity"]["high_usd"] else None)
        # Provider gross inconsistent with original x provider_fx in cases not flagged above
        elif abs(r.implied_gross - r.gross_amount_usd) > max(0.1, 0.002 * abs(r.gross_amount_usd)):
            F.add("PROVIDER_CALC_ERROR", txn=r.transaction_id, prov=r.provider_record_id,
                  impact=r.gross_amount_usd - r.implied_gross,
                  detail=f"{r.original_amount}×{r.provider_fx_rate_to_usd}={r.implied_gross} ≠ gross {r.gross_amount_usd}")

        if r.merchant_country != r.merchant_country_p:
            F.add("MERCHANT_COUNTRY_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id,
                  detail=f"{r.merchant_country} vs {r.merchant_country_p}")

        # net = gross - fee (purchase) / gross + fee (refund: fee is returned). Sign is taken from gross.
        exp_net = r.gross_amount_usd + r.provider_fee_usd if (r.gross_amount_usd < 0 and FEE["refund_fee_returned"]) \
            else r.gross_amount_usd - r.provider_fee_usd
        if abs(r.net_amount_usd - exp_net) > cfg["net"]["abs_tolerance_usd"]:
            F.add("NET_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id, impact=r.net_amount_usd - exp_net,
                  detail=f"net {r.net_amount_usd} vs expected {exp_net:.2f} (gross {r.gross_amount_usd}, fee {r.provider_fee_usd})")

        # Fee: a tier fits provider gross OR internal gross (in the latter case the gross is wrong, not the fee)
        fits_p = _fee_fits(r.provider_fee_usd, r.gross_amount_usd, FEE)
        fits_i = _fee_fits(r.provider_fee_usd, r.internal_gross_usd, FEE)
        if not (fits_p or fits_i):
            g = abs(r.gross_amount_usd)
            nearest = min((max(FEE["min_fee_usd"], round(t * g, 2)) for t in FEE["tiers"]),
                          key=lambda e: abs(e - r.provider_fee_usd))
            F.add("FEE_ANOMALY", txn=r.transaction_id, prov=r.provider_record_id, impact=r.provider_fee_usd - nearest,
                  detail=f"fee {r.provider_fee_usd} = {safe_div(r.provider_fee_usd, g):.2%} of {g}; expected ≈{nearest} "
                         f"(tiers {[f'{t:.1%}' for t in FEE['tiers']]}, min {FEE['min_fee_usd']})")

        if r.lag_h < 0:
            F.add("SETTLED_BEFORE_CREATED", txn=r.transaction_id, prov=r.provider_record_id,
                  detail=f"lag {r.lag_h:.0f}h")
        elif r.lag_h > T["late_settlement_hours"]:
            F.add("LATE_SETTLEMENT", txn=r.transaction_id, prov=r.provider_record_id,
                  detail=f"settled after {r.lag_h:.0f}h (median {mm.lag_h.median():.0f}h, p95 {mm.lag_h.quantile(.95):.0f}h)")

    # ---------- Possible duplicates inside the ledger (double charge) ----------
    comp = i[i.status == "COMPLETED"].sort_values("created_at_utc")
    grp = ["user_id", "original_currency", "original_amount", "merchant_id"]
    # groupby drops rows with a null key; count them (each is already reported as INVALID_VALUE)
    skipped_dup = int(comp[grp].isna().any(axis=1).sum())
    for _, g in comp.groupby(grp):
        if len(g) < 2:
            continue
        t = g.created_at_utc.values
        for a in range(1, len(g)):
            gap = (t[a] - t[a - 1]) / np.timedelta64(1, "m")
            if gap <= cfg["duplicates"]["window_minutes"]:
                r0, r1 = g.iloc[a - 1], g.iloc[a]
                both = all(x in set(p[kp]) for x in (r0.transaction_id, r1.transaction_id))
                F.add("POSSIBLE_DUPLICATE_INTERNAL", txn=r1.transaction_id, impact=r1.internal_gross_usd,
                      detail=f"{r0.transaction_id} and {r1.transaction_id}: user {r1.user_id}, merchant {r1.merchant_id}, "
                             f"{r1.original_amount} {r1.original_currency}, gap {gap:.0f} min; "
                             f"category {r0.merchant_category}/{r1.merchant_category}, channel {r0.channel}/{r1.channel}; "
                             f"both settled by provider: {'yes' if both else 'no'}")

    # ---------- Outliers: statistically rare != error ----------
    outl = []
    base = i[i.status == "COMPLETED"].copy()
    base["abs_usd"] = base.internal_gross_usd.abs()
    base["z"] = np.nan
    for t, g in base.groupby(base.transaction_type.fillna("UNKNOWN")):
        g = g[g.abs_usd > 0]
        base.loc[g.index, "z"] = robust_z(np.log(g.abs_usd))
    cand = base[(base.z.abs() > O["robust_z_threshold"]) | (base.abs_usd >= O["high_value_usd"])]
    other = F.frame()
    other = other[~other.code.isin(["HIGH_VALUE_OUTLIER", "OUTLIER_WITH_DEFECT", "IN_FLIGHT"])]
    for r in cand.itertuples():
        defects = other[other.transaction_id == r.transaction_id]
        defects = defects[defects.severity.isin(["HIGH", "MEDIUM"])]
        rec = m.loc[r.Index]
        if len(defects):
            verdict = "REAL_ISSUE"
            F.add("OUTLIER_WITH_DEFECT", txn=r.transaction_id,
                  impact=float(defects.impact_usd.sum()),
                  detail=f"outlier (z={r.z:.1f}, {r.abs_usd} USD) + defects: {', '.join(defects.code)}")
        else:
            verdict = "VALID_OUTLIER"
            F.add("HIGH_VALUE_OUTLIER", txn=r.transaction_id, impact=0,
                  detail=f"{r.abs_usd} USD, robust z={r.z:.1f}: amounts, FX, fee, net and timing are all normal in both sources; "
                         f"business review only (limits, KYC / merchant {r.merchant_id}, category {r.merchant_category})")
        outl.append({"transaction_id": r.transaction_id, "abs_usd": r.abs_usd, "robust_z": round(r.z, 2),
                     "channel": r.channel, "merchant_id": r.merchant_id, "category": r.merchant_category,
                     "reconciled": bool(rec.matched), "verdict": verdict,
                     "defects": ", ".join(defects.code) if len(defects) else ""})

    exceptions = F.frame()

    # ---------- Reconciliation bridge (provider total − internal COMPLETED total) ----------
    pairs = mm.copy()
    flagged_ids = lambda codes: set(exceptions[exceptions.code.isin(codes)].transaction_id)
    struct = flagged_ids(["SIGN_OR_TYPE_MISMATCH"])
    amt = flagged_ids(["AMOUNT_MISMATCH", "FX_MISMATCH"]) - struct
    d = pairs.gross_diff
    nan_rows = d.isna()  # pairs where an amount is null: cannot be compared, shown as their own bridge step
    invalid_val = float((pairs.gross_amount_usd.fillna(0) - pairs.internal_gross_usd.fillna(0))[nan_rows].sum())
    bridge = [
        ("Internal gross (COMPLETED)", float(i[i.status == "COMPLETED"].internal_gross_usd.sum())),
        ("− Missing settlements", float(-unsettled[unsettled.transaction_id.isin(flagged_ids(["MISSING_SETTLEMENT"]))].internal_gross_usd.sum())),
        ("− In flight (not yet due)", float(-unsettled[~unsettled.transaction_id.isin(flagged_ids(["MISSING_SETTLEMENT"]))].internal_gross_usd.sum()) + 0.0),
        ("+ Orphan settlements", float(orphans.gross_amount_usd.sum())),
        ("+ Duplicate settlements", float(extras.gross_amount_usd.sum())),
        ("+ DECLINED settled", float(dec.gross_amount_usd.sum())),
        ("± Sign/type mismatch", float(d[pairs.transaction_id.isin(struct)].sum())),
        ("± Amount/FX mismatch", float(d[pairs.transaction_id.isin(amt)].sum())),
        ("± Rows with null amounts (not comparable)", invalid_val),
        ("± Normal FX noise (remainder)", float(d[~pairs.transaction_id.isin(struct | amt) & ~nan_rows].sum())),
    ]
    prov_total = float(p.gross_amount_usd.sum())
    bridge.append(("= Provider gross (all rows)", prov_total))
    assert abs(sum(v for _, v in bridge[:-1]) - prov_total) < 0.01, "bridge does not reconcile"
    bridge_df = pd.DataFrame(bridge, columns=["step", "usd"]).round(2)

    stats = {
        "internal_rows": len(i), "provider_rows": len(p),
        "internal_completed": int((i.status == "COMPLETED").sum()), "internal_declined": int((i.status == "DECLINED").sum()),
        "matched_pairs": int(m.matched.sum()), "provider_data_end": str(data_end),
        "provider_fees_usd": float(p.provider_fee_usd.sum()),
        "fee_rate_low_tier_share": float(((safe_div(mm.provider_fee_usd, mm.gross_amount_usd.abs()) < 0.0125) & (mm.gross_amount_usd.abs() > 3)).mean()),
        "settlement_lag_h_median": float(mm.lag_h.median()), "settlement_lag_h_p95": float(mm.lag_h.quantile(.95)),
        "gross_diff_noise_p95_usd": float(mm.gross_diff.abs().quantile(.95)),
        **guard_stats, "rows_skipped_in_duplicate_check": skipped_dup,
    }
    return exceptions, pd.DataFrame(outl), bridge_df, m, mm, stats
