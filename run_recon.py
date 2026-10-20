#!/usr/bin/env python3
"""Запуск: python3 run_recon.py [--config config.yaml] [--input file.xlsx] [--out output]

Повністю детермінований: однаковий вхід + конфіг => однаковий вихід. Код виходу 1,
якщо знайдено HIGH-виключення (зручно для CI/cron-перевірок).
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from recon.checks import run_checks          # noqa: E402
from recon.clean import load_and_clean       # noqa: E402
from recon.report import write_outputs       # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--input")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    base = Path(a.config).parent
    cfg = yaml.safe_load(Path(a.config).read_text())
    path = Path(a.input or cfg["input"]["path"])
    path = path if path.is_absolute() else base / path
    out = Path(a.out or cfg["output_dir"])
    out = out if out.is_absolute() else base / out

    i, p, clean_log = load_and_clean(str(path), cfg["input"]["internal_sheet"], cfg["input"]["provider_sheet"])
    exceptions, outliers, bridge, m, mm, stats = run_checks(i, p, cfg)
    by_code = write_outputs(out, exceptions, outliers, bridge, m, mm, clean_log, stats)

    print(f"internal={stats['internal_rows']} provider={stats['provider_rows']} matched={stats['matched_pairs']}")
    print(by_code.to_string(index=False))
    print("\nBridge:\n" + bridge.to_string(index=False))
    print(f"\nВихід: {out}")
    return 1 if (exceptions.severity == "HIGH").any() else 0


if __name__ == "__main__":
    sys.exit(main())
