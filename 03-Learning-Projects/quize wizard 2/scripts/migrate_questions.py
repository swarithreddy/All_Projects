#!/usr/bin/env python3
"""Migrate legacy V1 *.txt question banks to JSON."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiz_wizard.repositories.migrator import migrate_legacy_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        default=ROOT / "legacy",
        help="Directory containing V1 .txt banks",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "questions",
        help="Output directory for JSON banks",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    written = migrate_legacy_directory(args.legacy, args.out)
    print(f"Wrote {len(written)} question banks to {args.out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
