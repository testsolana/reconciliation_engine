# Financial Anomaly & Reconciliation Engine (прототип)

Звіряє internal ledger (`internal_transactions`, 320 рядків) із provider settlement (`provider_settlement`, 296 рядків).

## Запуск
```bash
pip install pandas openpyxl pyyaml
python3 run_recon.py                      # дані та пороги беруться з config.yaml
python3 run_recon.py --input new.xlsx --out out_2026_10   # інший файл
python3 -m unittest discover -s tests     # регресійні тести детекторів
```
Детермінований (однаковий вхід → побайтово той самий `exceptions.csv`). Код виходу `1`, якщо є HIGH-знахідки → можна ставити в cron/CI.
Результат у `output/`: `reconciliation_report.xlsx` (Summary, Findings_by_type, Exceptions, Recon_bridge, Outlier_review, Matched_pairs, Cleaning_log), `exceptions.csv`, `summary.json`.

## Архітектура
`clean.py` (схема, trim/upper, числа/дати, cleaning log) → `checks.py` (match по `transaction_id = client_reference`, детектори, outlier-класифікація, bridge) → `report.py`. Усі пороги — в `config.yaml`.

## Що з'ясовано про дані
- Ключ звірки: `transaction_id` ↔ `client_reference`. Усі суми провайдера в USD; `provider_fx_rate_to_usd` — курс **початкової** валюти (для перевірки `original_amount × fx = gross`).
- Дані формально чисті (типи, пробіли, NULL — без правок), тож проблеми — у змісті, а не в форматі.
- DECLINED (25) не мають settle-итись — норма: 24 з 25 відсутні в provider, 1 — ні (знахідка).
- **Конвенції провайдера, які НЕ є помилками:** (1) fee = 1.8% з мінімумом $0.05 (тому на малих чеках % вищий); (2) для refund `net = gross + fee` (fee повертається) — 18 з 19 refund, виняток — T202609-0085 з інвертованим знаком; (3) розбіжність gross до ±0.1% — FX-шум (р95 ≈ $0.09); (4) settlement через 2–70 год.
- **Відкрите питання до провайдера:** ~16% операцій мають fee ≈1.0% замість 1.8% (співвідношення стабільно ≈0.55, не пояснюється мерчантом/каналом/валютою/датою). Прийнято як другий тарифний тир; потрібно підтвердити з fee schedule — інакше це системне недоотримання/переплата.

## Знахідки (provider − internal, USD)
| Sev | Тип | Записи | Exposure |
|---|---|---|---|
| HIGH | Знак refund інвертований (ledger −160.08, provider +160.03) | T202609-0085 | 320.11 |
| HIGH | Подвійний settlement (P00041 = P00296) | T202609-0042 | 217.15 |
| HIGH | Orphan settlement без ledger-запису | T202609-9999 / P00297 | 146.80 |
| HIGH | COMPLETED без settlement (вік >150h) | T202609-0011, -0174 | 34.62 |
| HIGH | DECLINED, але SETTLED | T202609-0206 | 19.49 |
| HIGH | Purchase розрахований як REFUND | T202609-0115 | до 42.84 (знак тип↔сума) |
| MED | Можливий double-charge (той самий user/merchant/сума, 2 хв) — обидва settled | T202609-0061/0062 | 54.18 |
| MED | Сума провайдера 14.69 vs 2.19 (не виводиться з FX) | T202609-0077 | 12.50 |
| MED | FX провайдера ±6–7% від ринку валюти | T202609-0125, -0212 | 4.13 |
| MED | net ≠ gross − fee | T202609-0275 | 3.33 |
| MED | Fee поза тирами (6.5% і 0.12%) | T202609-0188, -0248 | 2.49 |
| LOW | Settlement через 268h (норма ≤70h) | T202609-0264 | 0 |
| INFO | Outlier $8,900 — звівся в обох системах, fee 1.8%, FX у нормі | T202609-0025 | 0 |

Bridge: internal COMPLETED 17,208.18 → provider 17,886.83 (різниця +678.65 повністю розкладена по категоріях, у `Recon_bridge`).

## Реальна проблема vs нормальний outlier
Правило: статистична рідкість (modified z-score на log|USD|, окремо purchase/refund, поріг 3.5, + policy-поріг $1000) **сама по собі не є помилкою**. Outlier → `HIGH_VALUE_OUTLIER` (INFO, лише бізнес-перегляд), якщо суми, FX, fee, net, час зійшлись у двох джерелах; → `OUTLIER_WITH_DEFECT` (HIGH), якщо є хоч один дефект reconciliation. Аналогічно: високий % fee на дрібних чеках — наслідок мінімуму $0.05, а не аномалія; gross-різниця $4.27 на $8,900 (0.05%) — шум, а $2.78 на $40 (7%) — реальний FX-збій. Тобто **відносна** міра, прив'язана до моделі ціноутворення, а не абсолютна сума.

## Які ще аномалії варто перевіряти (не реалізовано/потребує даних)
- Refund без пов'язаної оригінальної покупки: у ledger немає `original_transaction_id`; жоден з 19 refund не має покупки того ж user+merchant → додати поле й перевірку refund ≤ original.
- Velocity/структурування: багато операцій одного user/card за короткий час, суми трохи нижче лімітів, card_country ≠ merchant_country + високий чек.
- Агрегація по batch/дню: сума batch у провайдера vs сума його рядків та vs банківська виписка (payout).
- Дрейф fee (середній % по днях/мерчантах), зміна тарифного тиру; нові валюти/курси; застарілі курси (stale FX) у вихідні.
- `merchant_id` зустрічається у 3–8 країнах — або довідник мерчантів некоректний, або країна не належить мерчанту; перевірити master data.
- Settlement у вихідні/поза вікном, перетин батчів, повторне використання `provider_record_id`; late-arriving записи у наступних звітах (дедуплікація між періодами).
- Ефект in-flight: при запуску на неповному періоді не рахувати як missing (`in_flight_grace_hours`).

## Обмеження
Датасет синтетичний і малий (≈300 рядків): пороги підібрані за його розподілом (FX-шум ±0.12%, fee-тири), на продуктивних даних їх треба перекалібрувати в `config.yaml`. Тири fee виведені з даних, а не з контракту.
