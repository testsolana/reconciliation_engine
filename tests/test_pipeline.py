"""Regression tests: known problems are injected into a clean synthetic dataset and the detectors must catch them,
while normal noise / clean outliers must not be flagged.  Run: python3 -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recon.checks import run_checks  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RATE = {"USD": 1.0, "EUR": 1.085}


def make(n=60):
    rows_i, rows_p = [], []
    for k in range(n):
        cur = "USD" if k % 2 else "EUR"
        amt = round(10 + (k * 7) % 90 + 0.37, 2)
        gross = round(amt * RATE[cur], 2)
        t = pd.Timestamp("2026-09-01") + pd.Timedelta(hours=3 * k + 1, minutes=k)
        tid = f"T{k:04d}"
        rows_i.append(dict(transaction_id=tid, user_id=f"U{k:03d}", created_at_utc=t, transaction_type="PURCHASE",
                           status="COMPLETED", original_currency=cur, original_amount=amt,
                           internal_fx_rate_to_usd=RATE[cur], internal_gross_usd=gross, card_country="DE",
                           merchant_country="DE", merchant_id=f"M{k % 9:03d}", merchant_category="5411", channel="ECOM"))
        fee = max(0.05, round(0.018 * gross, 2))
        st = t + pd.Timedelta(hours=30)
        rows_p.append(dict(provider_record_id=f"P{k:04d}", client_reference=tid, settled_at_utc=st,
                           settlement_type="PURCHASE", provider_status="SETTLED", settlement_currency="USD",
                           gross_amount_usd=gross, provider_fee_usd=fee, net_amount_usd=round(gross - fee, 2),
                           provider_fx_rate_to_usd=RATE[cur], merchant_country="DE", batch_id=st.strftime("B%Y%m%d")))
    # anchor row so the provider data end is far from the "in flight" window
    last = rows_p[-1].copy(); last["provider_record_id"] = "P9999"; last["client_reference"] = "TX-ANCHOR"
    last["settled_at_utc"] = pd.Timestamp("2026-09-20"); last["batch_id"] = "B20260920"
    rows_p.append(last)
    return pd.DataFrame(rows_i), pd.DataFrame(rows_p)


def codes(exc, txn):
    return set(exc[exc.transaction_id == txn].code)


class PipelineTests(unittest.TestCase):
    def test_clean_data_has_only_anchor_orphan(self):
        i, p = make()
        exc = run_checks(i, p, CFG)[0]
        self.assertEqual(set(exc.code), {"ORPHAN_SETTLEMENT"})

    def test_injected_issues_detected(self):
        i, p = make()
        p = pd.concat([p, p[p.client_reference == "T0003"]], ignore_index=True)           # duplicate settlement
        p.loc[p.client_reference == "T0004", "gross_amount_usd"] += 25                      # amount
        p.loc[p.client_reference == "T0005", "provider_fee_usd"] = 5.0                      # fee
        p.loc[p.client_reference == "T0006", "net_amount_usd"] -= 3                         # net
        p.loc[p.client_reference == "T0007", "settled_at_utc"] += pd.Timedelta(hours=200)   # late
        p = p[p.client_reference != "T0008"]                                                # missing
        i.loc[i.transaction_id == "T0009", "status"] = "DECLINED"                           # declined settled
        i.loc[i.transaction_id == "T0010", ["transaction_type", "original_amount", "internal_gross_usd"]] = \
            ["REFUND", -50.0, -50.0]
        p.loc[p.client_reference == "T0010", ["settlement_type", "gross_amount_usd", "net_amount_usd"]] = \
            ["REFUND", 50.0, 49.1]                                                          # sign flip
        i = pd.concat([i, i[i.transaction_id == "T0011"].assign(
            transaction_id="T0011B", created_at_utc=i.loc[i.transaction_id == "T0011", "created_at_utc"] + pd.Timedelta(minutes=2))])
        exc = run_checks(i, p, CFG)[0]
        self.assertIn("DUPLICATE_SETTLEMENT", codes(exc, "T0003"))
        self.assertIn("AMOUNT_MISMATCH", codes(exc, "T0004"))
        self.assertIn("FEE_ANOMALY", codes(exc, "T0005"))
        self.assertIn("NET_MISMATCH", codes(exc, "T0006"))
        self.assertIn("LATE_SETTLEMENT", codes(exc, "T0007"))
        self.assertIn("MISSING_SETTLEMENT", codes(exc, "T0008"))
        self.assertIn("DECLINED_SETTLED", codes(exc, "T0009"))
        self.assertIn("SIGN_OR_TYPE_MISMATCH", codes(exc, "T0010"))
        self.assertIn("POSSIBLE_DUPLICATE_INTERNAL", codes(exc, "T0011B"))

    def test_fx_shift_detected_once(self):
        i, p = make()
        p.loc[p.client_reference == "T0001", "provider_fx_rate_to_usd"] = 1.07
        p.loc[p.client_reference == "T0001", "gross_amount_usd"] = round(i.loc[1, "original_amount"] * 1.07, 2)
        exc = run_checks(i, p, CFG)[0]
        self.assertIn("FX_MISMATCH", codes(exc, "T0001"))
        self.assertNotIn("AMOUNT_MISMATCH", codes(exc, "T0001"))

    def test_normal_fx_noise_not_flagged(self):
        i, p = make()
        for k in range(0, 40, 2):
            p.loc[p.client_reference == f"T{k:04d}", "provider_fx_rate_to_usd"] *= 1.001
            p.loc[p.client_reference == f"T{k:04d}", "gross_amount_usd"] += 0.05
            p.loc[p.client_reference == f"T{k:04d}", "net_amount_usd"] += 0.05
        exc = run_checks(i, p, CFG)[0]
        self.assertEqual(set(exc.code), {"ORPHAN_SETTLEMENT"})

    def test_clean_outlier_is_info_not_issue(self):
        i, p = make()
        i.loc[i.transaction_id == "T0020", ["original_currency", "original_amount", "internal_fx_rate_to_usd", "internal_gross_usd"]] = ["USD", 9000.0, 1.0, 9000.0]
        p.loc[p.client_reference == "T0020", ["gross_amount_usd", "provider_fee_usd", "net_amount_usd", "provider_fx_rate_to_usd"]] = [8996.0, 161.93, 8834.07, 1.0]
        exc = run_checks(i, p, CFG)[0]
        self.assertEqual(codes(exc, "T0020"), {"HIGH_VALUE_OUTLIER"})

    def test_outlier_with_defect_escalates(self):
        i, p = make()
        i.loc[i.transaction_id == "T0020", ["original_currency", "original_amount", "internal_fx_rate_to_usd", "internal_gross_usd"]] = ["USD", 9000.0, 1.0, 9000.0]
        p.loc[p.client_reference == "T0020", ["gross_amount_usd", "provider_fee_usd", "net_amount_usd", "provider_fx_rate_to_usd"]] = [8000.0, 144.0, 7856.0, 1.0]
        exc = run_checks(i, p, CFG)[0]
        self.assertIn("OUTLIER_WITH_DEFECT", codes(exc, "T0020"))

    def test_bridge_reconciles(self):
        i, p = make()
        p = p[p.client_reference != "T0008"]
        _, _, bridge, *_ = run_checks(i, p, CFG)
        self.assertAlmostEqual(bridge.usd.iloc[:-1].sum(), bridge.usd.iloc[-1], places=2)

    def test_deterministic(self):
        i, p = make()
        a = run_checks(i, p, CFG)[0]
        b = run_checks(i, p, CFG)[0]
        pd.testing.assert_frame_equal(a, b)


    # ---- robustness: zero / null values and null keys must be reported, never silent or crashing ----
    def test_safe_div(self):
        import numpy as np
        from recon.checks import safe_div
        self.assertTrue(np.isnan(safe_div(1.0, 0.0)))
        self.assertTrue(np.isnan(safe_div(1.0, float("nan"))))
        self.assertAlmostEqual(safe_div(1.0, 4.0), 0.25)
        out = safe_div(pd.Series([1.0, 2.0]), pd.Series([0.0, 4.0]))
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertEqual(out.iloc[1], 0.5)

    def test_zero_and_null_values_are_reported_not_silent(self):
        i, p = make()
        i.loc[i.transaction_id == "T0004", "internal_fx_rate_to_usd"] = 0.0     # zero rate -> would be inf
        p.loc[p.client_reference == "T0012", "gross_amount_usd"] = float("nan")  # NaN comparisons are always False
        p.loc[p.client_reference == "T0013", "gross_amount_usd"] = 0.0           # zero gross
        exc, _, bridge, *_ = run_checks(i, p, CFG)
        for t in ("T0004", "T0012", "T0013"):
            self.assertIn("INVALID_VALUE", codes(exc, t))
        self.assertNotIn("SIGN_OR_TYPE_MISMATCH", codes(exc, "T0012"))          # no fake mismatch from NaN
        self.assertNotIn("SIGN_OR_TYPE_MISMATCH", codes(exc, "T0013"))          # zero is not a sign flip
        self.assertAlmostEqual(bridge.usd.iloc[:-1].sum(), bridge.usd.iloc[-1], places=2)

    def test_null_keys_are_reported_not_dropped(self):
        i, p = make()
        p.loc[p.client_reference == "T0006", "client_reference"] = float("nan")
        i.loc[i.transaction_id == "T0010", "transaction_id"] = float("nan")
        exc, _, bridge, m, _, stats = run_checks(i, p, CFG)
        self.assertEqual(int((exc.code == "NULL_KEY").sum()), 2)
        self.assertEqual(stats["rows_with_null_keys"], 2)
        self.assertIn("MISSING_SETTLEMENT", codes(exc, "T0006"))                 # ledger side lost its partner
        self.assertEqual(len(m), len(i))                                         # no row lost or multiplied
        self.assertAlmostEqual(bridge.usd.iloc[:-1].sum(), bridge.usd.iloc[-1], places=2)
        # the two null keys must not be merged with each other
        self.assertFalse(m[m.transaction_id.str.startswith("__NULL_I")].matched.any())

    def test_null_group_keys_are_counted(self):
        i, p = make()
        i.loc[i.transaction_id.isin(["T0007", "T0008"]), "user_id"] = float("nan")
        exc, _, _, _, _, stats = run_checks(i, p, CFG)
        self.assertEqual(stats["rows_skipped_in_duplicate_check"], 2)
        self.assertEqual(int((exc.code == "INVALID_VALUE").sum()), 2)

    def test_null_transaction_type_does_not_crash_or_fake_mismatch(self):
        i, p = make()
        i.loc[i.transaction_id == "T0009", "transaction_type"] = float("nan")
        exc = run_checks(i, p, CFG)[0]
        self.assertIn("INVALID_VALUE", codes(exc, "T0009"))
        self.assertNotIn("SIGN_OR_TYPE_MISMATCH", codes(exc, "T0009"))


    def test_in_flight_transaction_keeps_bridge_balanced(self):
        i, p = make()
        created = i.loc[i.transaction_id == "T0059", "created_at_utc"].iloc[0]
        p = p[p.client_reference != "T0059"].copy()
        p.loc[p.provider_record_id == "P9999", "settled_at_utc"] = created + pd.Timedelta(hours=10)
        exc, _, bridge, *_ = run_checks(i, p, CFG)
        self.assertIn("IN_FLIGHT", codes(exc, "T0059"))
        self.assertNotIn("MISSING_SETTLEMENT", codes(exc, "T0059"))
        self.assertAlmostEqual(bridge.usd.iloc[:-1].sum(), bridge.usd.iloc[-1], places=2)


if __name__ == "__main__":
    unittest.main()
