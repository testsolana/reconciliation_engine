# Financial Anomaly & Reconciliation Engine (prototype)

Reconciles an internal transaction ledger (`internal_transactions`, 320 rows) against a provider settlement report (`provider_settlement`, 296 rows) and reports duplicates, missing operations, and amount / fee / FX discrepancies.

## start

**Requirements:** Python 3.10 or newer recommended (tested on 3.13). On Windows use `python` instead of `python3`.

**1. Unpack and enter the project folder** (all commands below are run from here):
```bash
unzip recon_engine.zip
cd recon_engine
```

**2. Install dependencies (once):**
```bash
pip install pandas openpyxl pyyaml
```
If `pip` is not found: `python3 -m pip install pandas openpyxl pyyaml`.

**3. Run the reconciliation:**
```bash
python3 run_recon.py
```
It prints a summary of findings by type and the reconciliation bridge, and writes the results to `output/`.

**4. Open the results** in `output/`:

| File | What it is |
|---|---|
| `reconciliation_report.xlsx` | Main report. Start with the **Exceptions** sheet (also: Summary, Findings_by_type, Recon_bridge, Outlier_review, Matched_pairs, Cleaning_log) |
| `exceptions.csv` | The same findings as CSV |
| `summary.json` | Statistics and bridge, machine-readable |

**Expected result on the included dataset:** `internal=320 provider=296 matched=294`, 16 findings (7 HIGH, 6 MEDIUM, 2 LOW, 1 INFO), and a bridge that reconciles from 17,208.18 to 17,886.83.

The process exits with code `1` when at least one HIGH finding exists. This is normal for this dataset, not an error; it lets cron/CI jobs raise an alert.

### Other commands
```bash
python3 run_recon.py --input my_file.xlsx --out my_results   # another input file / output folder
python3 run_recon.py --config my_config.yaml                 # another set of thresholds
python3 -m unittest discover -s tests                        # 18 regression tests, expected: OK
```
The input workbook must have the same two sheets and columns as the included one (`internal_transactions`, `provider_settlement`); sheet names and all thresholds are set in `config.yaml`.

The run is deterministic: the same input and config always produce a byte-identical `exceptions.csv`. Nothing is deleted or modified in the input data; suspicious rows are only flagged and recorded in the output files.

## Architecture
`clean.py` (schema check, trim/upper, numbers/dates, cleaning log) -> `checks.py` (match on `transaction_id = client_reference`, detectors, outlier classification, bridge) -> `report.py`. All thresholds are in `config.yaml`.

## What the data looks like
- Join key: `transaction_id` <-> `client_reference`. All provider amounts are in USD; `provider_fx_rate_to_usd` is the rate of the **original** currency (used to check `original_amount x fx = gross`).
- The data was formally clean (types, whitespace, NULLs needed no fixes), so the problems are in the content, not the format.
- DECLINED transactions (25) are not expected to settle: 24 of 25 are absent from the provider report, 1 is not (a finding).
- **Provider conventions that are NOT errors:** (1) fee = 1.8% with a $0.05 minimum (so small tickets show a higher %); (2) for refunds `net = gross + fee` (fee is returned) - 18 of 19 refunds, the exception being T202609-0085 with an inverted sign; (3) gross differences up to ±0.1% are FX noise (p95 ≈ $0.09); (4) settlement within 2-70 hours.
- **Open question for the provider:** ~16% of transactions carry a fee of ≈1.0% instead of 1.8% (ratio consistently ≈0.55, not explained by merchant/channel/currency/date). Treated as a second fee tier; confirm against the fee schedule, otherwise it is a systematic under/over-charge.

## Findings (provider - internal, USD)
| Severity | Type | Records | Exposure |
|---|---|---|---|
| HIGH | Refund sign inverted (ledger -160.08, provider +160.03) | T202609-0085 | 320.11 |
| HIGH | Duplicate settlement (P00041 = P00296) | T202609-0042 | 217.15 |
| HIGH | Orphan settlement with no ledger record | T202609-9999 / P00297 | 146.80 |
| HIGH | COMPLETED without settlement (age >150h) | T202609-0011, -0174 | 34.62 |
| HIGH | DECLINED but SETTLED | T202609-0206 | 19.49 |
| HIGH | Purchase settled as REFUND | T202609-0115 | up to 42.84 (type vs sign) |
| LOW | Possible double charge (same user/merchant/amount, 2 min apart, both settled); low confidence because channel and category differ | T202609-0061 / 0062 | 54.18 |
| MEDIUM | Provider amount 14.69 vs ledger 2.19 (not explained by FX) | T202609-0077 | 12.50 |
| MEDIUM | Provider FX rate 6-7% off the currency's normal range | T202609-0125, -0212 | 4.13 |
| MEDIUM | net != gross - fee | T202609-0275 | 3.33 |
| MEDIUM | Fee outside any tier (6.5% and 0.12%) | T202609-0188, -0248 | 2.49 |
| LOW | Settlement after 268h (normal <=70h) | T202609-0264 | 0 |
| INFO | $8,900 outlier - reconciles in both systems, fee 1.8%, FX normal | T202609-0025 | 0 |

Bridge: internal COMPLETED 17,208.18 -> provider 17,886.83. The +678.65 difference is fully decomposed by category (see `Recon_bridge`); the pipeline asserts that it reconciles.

## Duplicate confidence
A pair of near-identical ledger operations is only a *signal*. Confidence is scored from the attributes a technical double-submit would repeat: same channel and same merchant category = **high** (severity HIGH if both rows were settled, otherwise MEDIUM); one of them = **medium** (MEDIUM); neither = **low** (LOW). The mapping is configurable in `config.yaml` (`duplicates.confidence_severity`). T202609-0061/0062 differ in both, so it is LOW.

## Real problem vs. normal outlier
Rule: statistical rarity (modified z-score on log|USD|, computed separately for purchases and refunds, threshold 3.5, plus a $1,000 policy threshold) is **not an error by itself**. An outlier becomes `HIGH_VALUE_OUTLIER` (INFO, business review only) if amounts, FX, fee, net and timing agree in both sources, and `OUTLIER_WITH_DEFECT` (HIGH) if it has at least one reconciliation defect. Likewise: a high fee % on tiny tickets follows from the $0.05 minimum, not an anomaly; a $4.27 gross difference on $8,900 (0.05%) is noise, while $2.78 on $40 (7%) is a real FX failure. The measure is relative and tied to the pricing model, not an absolute amount.

## Robustness to dirty data
- **No silent skips.** Null/zero/non-positive values in fields the checks rely on are reported as `INVALID_VALUE`; empty keys are reported as `NULL_KEY`. (A NaN comparison is always False, so such rows would otherwise pass every check.)
- **No lost rows.** Empty keys get a unique placeholder (`__NULL_I<row>` / `__NULL_P<row>`), so they are never merged with each other and never dropped by a `groupby`; they surface as missing settlements or orphans. A row-accounting check raises a descriptive error if matching ever loses or multiplies a row.
- **Safe division.** All ratios go through `safe_div`, which returns NaN instead of `inf` / `ZeroDivisionError`.
- **Bridge stays balanced** with in-flight transactions and rows that have null amounts (own bridge steps).
- Rows skipped by the duplicate check because of a null grouping field are counted in `rows_skipped_in_duplicate_check`.

## Limitations
The dataset is synthetic and small (~300 rows): thresholds are tuned to its distribution (FX noise ±0.12%, fee tiers) and must be recalibrated in `config.yaml` for production data. Fee tiers are inferred from the data, not from a contract.
