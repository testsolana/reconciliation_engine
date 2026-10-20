"""Matching + усі детектори аномалій.

Конвенція impact_usd: provider − internal (знак = на скільки провайдер "більше" за наш облік).
Для MISSING_SETTLEMENT impact = −internal gross (гроші, які очікуємо, але не отримали).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# code -> (категорія, базова severity, пояснення "що це означає для бізнесу")
CATALOG = {
    "DUPLICATE_SETTLEMENT": ("Reconciliation", "HIGH", "Один платіж розрахований двічі: ризик подвійного списання/виплати."),
    "ORPHAN_SETTLEMENT": ("Reconciliation", "HIGH", "Провайдер розрахував операцію, якої немає в ledger: ризик фрод/збою обліку."),
    "DECLINED_SETTLED": ("Reconciliation", "HIGH", "Відхилена операція пройшла settlement: гроші списано без валідної транзакції."),
    "MISSING_SETTLEMENT": ("Reconciliation", "HIGH", "COMPLETED операція не розрахована провайдером: недоотримані кошти."),
    "SIGN_OR_TYPE_MISMATCH": ("Reconciliation", "HIGH", "Невідповідність типу/знаку (purchase↔refund): ефект на баланс = 2× суми."),
    "AMOUNT_MISMATCH": ("Reconciliation", "MEDIUM", "Gross у провайдера ≠ ledger поза допуском FX-шуму."),
    "FX_MISMATCH": ("FX", "MEDIUM", "Курс провайдера відхиляється від ledger і від ринкового діапазону валюти."),
    "FX_INTERNAL_OUT_OF_BAND": ("FX", "MEDIUM", "Внутрішній курс поза нормальним діапазоном валюти."),
    "FEE_ANOMALY": ("Fee", "MEDIUM", "Fee не відповідає жодному відомому тарифному тиру/мінімуму."),
    "NET_MISMATCH": ("Fee", "MEDIUM", "net ≠ gross ∓ fee: арифметична помилка у провайдера."),
    "LATE_SETTLEMENT": ("Timing", "LOW", "Settlement значно пізніше за норму: cash-flow та ризик втрати запису."),
    "SETTLED_BEFORE_CREATED": ("Timing", "MEDIUM", "Settlement раніше створення транзакції: проблема з часом/ідентифікатором."),
    "POSSIBLE_DUPLICATE_INTERNAL": ("Duplicates", "MEDIUM", "Схожі операції одного клієнта за хвилини: можливий double-charge."),
    "MERCHANT_COUNTRY_MISMATCH": ("Reconciliation", "LOW", "Країна мерчанта різна у двох системах."),
    "INTERNAL_CALC_ERROR": ("Data quality", "MEDIUM", "original_amount × fx ≠ internal_gross_usd."),
    "INTERNAL_SIGN_ERROR": ("Data quality", "MEDIUM", "Знак суми не відповідає типу операції (purchase>0, refund<0)."),
    "PROVIDER_CALC_ERROR": ("Data quality", "MEDIUM", "gross провайдера ≠ original_amount × provider_fx."),
    "BATCH_DATE_MISMATCH": ("Data quality", "LOW", "batch_id не відповідає даті settlement."),
    "DUPLICATE_KEY": ("Data quality", "HIGH", "Дубльований ключ (transaction_id/provider_record_id)."),
    "HIGH_VALUE_OUTLIER": ("Outlier", "INFO", "Статистичний outlier, що звівся без розбіжностей: нормальна, але потребує business review."),
    "OUTLIER_WITH_DEFECT": ("Outlier", "HIGH", "Outlier, який ще й має дефект reconciliation: пріоритетний розбір."),
    "IN_FLIGHT": ("Reconciliation", "INFO", "COMPLETED без settlement, але ще в межах нормального вікна."),
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
    """Modified z-score (Iglewicz–Hoaglin): стійкий до самих outliers."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


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
    i, p = i.reset_index(drop=True), p.reset_index(drop=True)  # код нижче покладається на унікальний індекс
    m, extras, orphans, p_sorted = build_match(i, p, cfg)
    ki, kp = cfg["matching"]["key_internal"], cfg["matching"]["key_provider"]
    A, FX, FEE, T, O = cfg["amount"], cfg["fx"], cfg["fee"], cfg["timing"], cfg["outliers"]

    # ---------- Data quality: внутрішній ledger ----------
    for k in i.loc[i[ki].duplicated(keep=False), ki].unique():
        F.add("DUPLICATE_KEY", txn=k, detail=f"transaction_id {k} зустрічається {int((i[ki] == k).sum())} разів")
    for k in p.loc[p.provider_record_id.duplicated(keep=False), "provider_record_id"].unique():
        F.add("DUPLICATE_KEY", prov=k, detail=f"provider_record_id {k} повторюється")

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
              detail=f"{r.transaction_type} з сумою {r.original_amount}")

    # FX band для internal курсів (по валюті, MAD-z)
    med_rate = i.groupby("original_currency").internal_fx_rate_to_usd.median()
    i_dev = i.internal_fx_rate_to_usd / i.original_currency.map(med_rate) - 1
    for r in i[(i_dev.abs() > FX["band_min_rel"])].itertuples():
        F.add("FX_INTERNAL_OUT_OF_BAND", txn=r.transaction_id,
              detail=f"{r.original_currency} rate {r.internal_fx_rate_to_usd} vs медіана {med_rate[r.original_currency]:.4f} ({i_dev[r.Index]:+.2%})")

    # ---------- Data quality: provider ----------
    for r in p_sorted[p_sorted.settled_at_utc.dt.strftime("B%Y%m%d") != p_sorted.batch_id].itertuples():
        F.add("BATCH_DATE_MISMATCH", prov=r.provider_record_id, txn=r.client_reference,
              detail=f"settled {r.settled_at_utc:%Y-%m-%d}, batch {r.batch_id}")

    # ---------- Reconciliation: структурні ----------
    for r in extras.itertuples():
        first = p_sorted[(p_sorted[kp] == r.client_reference) & (p_sorted._occ == 0)].iloc[0]
        same = (first.gross_amount_usd == r.gross_amount_usd and first.settled_at_utc == r.settled_at_utc)
        F.add("DUPLICATE_SETTLEMENT", txn=r.client_reference, prov=r.provider_record_id, impact=r.gross_amount_usd,
              detail=f"{r.provider_record_id} повторює {first.provider_record_id} "
                     f"({'ідентичні суми та час' if same else 'різні суми/час'}); зайвий gross {r.gross_amount_usd}")
    for r in orphans.itertuples():
        F.add("ORPHAN_SETTLEMENT", txn=r.client_reference, prov=r.provider_record_id, impact=r.gross_amount_usd,
              detail=f"client_reference {r.client_reference} відсутній у ledger; gross {r.gross_amount_usd}, fee {r.provider_fee_usd}")

    dec = m[m.matched & (m.status == "DECLINED")]
    for r in dec.itertuples():
        F.add("DECLINED_SETTLED", txn=r.transaction_id, prov=r.provider_record_id, impact=r.gross_amount_usd,
              detail=f"internal status DECLINED, але settlement {r.provider_status} gross {r.gross_amount_usd}")

    data_end = p.settled_at_utc.max()
    unsettled = m[(~m.matched) & (m.status == "COMPLETED")]
    for r in unsettled.itertuples():
        age = (data_end - r.created_at_utc).total_seconds() / 3600
        if age > T["in_flight_grace_hours"]:
            F.add("MISSING_SETTLEMENT", txn=r.transaction_id, impact=-r.internal_gross_usd,
                  detail=f"COMPLETED {r.created_at_utc:%Y-%m-%d %H:%M}; у provider-звіті (до {data_end:%Y-%m-%d}) немає, вік {age:.0f}h")
        else:
            F.add("IN_FLIGHT", txn=r.transaction_id, impact=0,
                  detail=f"вік {age:.0f}h < {T['in_flight_grace_hours']}h")

    # ---------- Reconciliation: порівняння сум для matched COMPLETED ----------
    mm = m[m.matched & (m.status == "COMPLETED")].copy()
    mm["gross_diff"] = mm.gross_amount_usd - mm.internal_gross_usd
    mm["fx_rel_diff"] = mm.provider_fx_rate_to_usd / mm.internal_fx_rate_to_usd - 1
    mm["tol"] = np.maximum(A["abs_tolerance_usd"], A["rel_tolerance"] * mm.internal_gross_usd.abs())
    mm["implied_gross"] = (mm.original_amount * mm.provider_fx_rate_to_usd).round(2)
    prov_dev = mm.provider_fx_rate_to_usd / mm.original_currency.map(med_rate) - 1
    mm["lag_h"] = (mm.settled_at_utc - mm.created_at_utc).dt.total_seconds() / 3600

    flagged_fx = set()
    for r in mm.itertuples():
        pdv = prov_dev[r.Index]
        if abs(r.fx_rel_diff) > FX["rel_tolerance_vs_internal"] and abs(pdv) > FX["band_min_rel"]:
            flagged_fx.add(r.transaction_id)
            F.add("FX_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id,
                  impact=r.gross_amount_usd - r.internal_gross_usd,
                  detail=f"{r.original_currency}: provider rate {r.provider_fx_rate_to_usd:.4f} vs internal "
                         f"{r.internal_fx_rate_to_usd:.4f} ({r.fx_rel_diff:+.2%}); провайдер {pdv:+.2%} від медіани валюти, "
                         f"internal {i_dev[r.Index]:+.2%} → відхиляється provider")

    for r in mm.itertuples():
        sign_flip = np.sign(r.gross_amount_usd) != np.sign(r.internal_gross_usd)
        type_diff = r.settlement_type != r.transaction_type
        if sign_flip or type_diff:
            F.add("SIGN_OR_TYPE_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id,
                  impact=r.gross_amount_usd - r.internal_gross_usd,
                  detail=f"ledger {r.transaction_type} {r.internal_gross_usd} vs provider {r.settlement_type} {r.gross_amount_usd}"
                         f"{' (знак інвертовано)' if sign_flip else ''}; зсув балансу {r.gross_amount_usd - r.internal_gross_usd:+.2f}"
                         f"{'' if sign_flip else '; якщо REFUND трактується як вихідний платіж — ризик до 2×' + str(abs(r.internal_gross_usd))}",
                  exposure=abs(r.gross_amount_usd - r.internal_gross_usd) if sign_flip else 2 * abs(r.internal_gross_usd))
        elif abs(r.gross_diff) > r.tol:
            if r.transaction_id in flagged_fx and abs(r.implied_gross - r.gross_amount_usd) <= 0.02:
                continue  # повністю пояснено FX_MISMATCH (той самий impact) — не дублюємо
            elif abs(r.implied_gross - r.gross_amount_usd) > 0.02:
                cause = (f"gross не виводиться з original×provider_fx (={r.implied_gross}) → помилка суми, не FX")
            else:
                cause = "FX в нормі"
            F.add("AMOUNT_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id, impact=r.gross_diff,
                  detail=f"ledger {r.internal_gross_usd} vs provider {r.gross_amount_usd} (Δ{r.gross_diff:+.2f}, допуск ±{r.tol:.2f}); {cause}",
                  severity="HIGH" if abs(r.gross_diff) >= cfg["severity"]["high_usd"] else None)
        # Не-FX розбіжність між original×provider_fx та gross для випадків, що не flagged вище
        elif abs(r.implied_gross - r.gross_amount_usd) > max(0.1, 0.002 * abs(r.gross_amount_usd)):
            F.add("PROVIDER_CALC_ERROR", txn=r.transaction_id, prov=r.provider_record_id,
                  impact=r.gross_amount_usd - r.implied_gross,
                  detail=f"{r.original_amount}×{r.provider_fx_rate_to_usd}={r.implied_gross} ≠ gross {r.gross_amount_usd}")

        if r.merchant_country != r.merchant_country_p:
            F.add("MERCHANT_COUNTRY_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id,
                  detail=f"{r.merchant_country} vs {r.merchant_country_p}")

        # net = gross − fee (purchase) / gross + fee (refund: fee повертається). Знак визначає за gross.
        exp_net = r.gross_amount_usd + r.provider_fee_usd if (r.gross_amount_usd < 0 and FEE["refund_fee_returned"]) \
            else r.gross_amount_usd - r.provider_fee_usd
        if abs(r.net_amount_usd - exp_net) > cfg["net"]["abs_tolerance_usd"]:
            F.add("NET_MISMATCH", txn=r.transaction_id, prov=r.provider_record_id, impact=r.net_amount_usd - exp_net,
                  detail=f"net {r.net_amount_usd} vs очікуваний {exp_net:.2f} (gross {r.gross_amount_usd}, fee {r.provider_fee_usd})")

        # Fee: підходить тир на gross провайдера АБО на internal gross (тоді причина в gross, а не fee)
        fits_p = _fee_fits(r.provider_fee_usd, r.gross_amount_usd, FEE)
        fits_i = _fee_fits(r.provider_fee_usd, r.internal_gross_usd, FEE)
        if not (fits_p or fits_i):
            g = abs(r.gross_amount_usd)
            nearest = min((max(FEE["min_fee_usd"], round(t * g, 2)) for t in FEE["tiers"]),
                          key=lambda e: abs(e - r.provider_fee_usd))
            F.add("FEE_ANOMALY", txn=r.transaction_id, prov=r.provider_record_id, impact=r.provider_fee_usd - nearest,
                  detail=f"fee {r.provider_fee_usd} = {r.provider_fee_usd / g:.2%} від {g}; очікувалось ≈{nearest} "
                         f"(тири {[f'{t:.1%}' for t in FEE['tiers']]}, мін {FEE['min_fee_usd']})")

        if r.lag_h < 0:
            F.add("SETTLED_BEFORE_CREATED", txn=r.transaction_id, prov=r.provider_record_id,
                  detail=f"lag {r.lag_h:.0f}h")
        elif r.lag_h > T["late_settlement_hours"]:
            F.add("LATE_SETTLEMENT", txn=r.transaction_id, prov=r.provider_record_id,
                  detail=f"settlement через {r.lag_h:.0f}h (медіана {mm.lag_h.median():.0f}h, p95 {mm.lag_h.quantile(.95):.0f}h)")

    # ---------- Можливі дублікати в internal (double-charge) ----------
    comp = i[i.status == "COMPLETED"].sort_values("created_at_utc")
    grp = ["user_id", "original_currency", "original_amount", "merchant_id"]
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
                      detail=f"{r0.transaction_id} і {r1.transaction_id}: user {r1.user_id}, merchant {r1.merchant_id}, "
                             f"{r1.original_amount} {r1.original_currency}, інтервал {gap:.0f} хв; "
                             f"category {r0.merchant_category}/{r1.merchant_category}, channel {r0.channel}/{r1.channel}; "
                             f"обидві розраховані провайдером: {'так' if both else 'ні'}")

    # ---------- Outliers: статистика ≠ помилка ----------
    outl = []
    base = i[i.status == "COMPLETED"].copy()
    base["abs_usd"] = base.internal_gross_usd.abs()
    base["z"] = np.nan
    for t, g in base.groupby("transaction_type"):
        g = g[g.abs_usd > 0]
        base.loc[g.index, "z"] = robust_z(np.log(g.abs_usd))
    cand = base[(base.z.abs() > O["robust_z_threshold"]) | (base.abs_usd >= O["high_value_usd"])]
    other = F.frame()
    other = other[~other.code.isin(["HIGH_VALUE_OUTLIER", "OUTLIER_WITH_DEFECT", "IN_FLIGHT"])]
    for r in cand.itertuples():
        defects = other[other.transaction_id == r.transaction_id]
        defects = defects[defects.severity.isin(["HIGH", "MEDIUM"])]
        rec = m.loc[r.Index]
        reasons = []
        if not rec.matched:
            reasons.append("не розрахована провайдером")
        if len(defects):
            verdict = "REAL_ISSUE"
            F.add("OUTLIER_WITH_DEFECT", txn=r.transaction_id,
                  impact=float(defects.impact_usd.sum()),
                  detail=f"outlier (z={r.z:.1f}, {r.abs_usd} USD) + дефекти: {', '.join(defects.code)}")
        else:
            verdict = "VALID_OUTLIER"
            F.add("HIGH_VALUE_OUTLIER", txn=r.transaction_id, impact=0,
                  detail=f"{r.abs_usd} USD, robust z={r.z:.1f}: суми, FX, fee, net, час — у нормі в обох джерелах; "
                         f"рекомендується лише бізнес-перевірка (ліміти, KYC/мерчант {r.merchant_id}, категорія {r.merchant_category})")
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
    bridge = [
        ("Internal gross (COMPLETED)", float(i[i.status == "COMPLETED"].internal_gross_usd.sum())),
        ("− Missing settlements", float(-unsettled[unsettled.transaction_id.isin(flagged_ids(["MISSING_SETTLEMENT"]))].internal_gross_usd.sum())),
        ("+ Orphan settlements", float(orphans.gross_amount_usd.sum())),
        ("+ Duplicate settlements", float(extras.gross_amount_usd.sum())),
        ("+ DECLINED settled", float(dec.gross_amount_usd.sum())),
        ("± Sign/type mismatch", float(d[pairs.transaction_id.isin(struct)].sum())),
        ("± Amount/FX mismatch", float(d[pairs.transaction_id.isin(amt)].sum())),
        ("± Нормальний FX-шум (решта)", float(d[~pairs.transaction_id.isin(struct | amt)].sum())),
    ]
    prov_total = float(p.gross_amount_usd.sum())
    bridge.append(("= Provider gross (всі рядки)", prov_total))
    assert abs(sum(v for _, v in bridge[:-1]) - prov_total) < 0.01, "bridge не зводиться"
    bridge_df = pd.DataFrame(bridge, columns=["step", "usd"]).round(2)

    stats = {
        "internal_rows": len(i), "provider_rows": len(p),
        "internal_completed": int((i.status == "COMPLETED").sum()), "internal_declined": int((i.status == "DECLINED").sum()),
        "matched_pairs": int(m.matched.sum()), "provider_data_end": str(data_end),
        "provider_fees_usd": float(p.provider_fee_usd.sum()),
        "fee_rate_low_tier_share": float(((mm.provider_fee_usd / mm.gross_amount_usd.abs() < 0.0125) & (mm.gross_amount_usd.abs() > 3)).mean()),
        "settlement_lag_h_median": float(mm.lag_h.median()), "settlement_lag_h_p95": float(mm.lag_h.quantile(.95)),
        "gross_diff_noise_p95_usd": float(mm.gross_diff.abs().quantile(.95)),
    }
    return exceptions, pd.DataFrame(outl), bridge_df, m, mm, stats
