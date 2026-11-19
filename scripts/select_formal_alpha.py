#!/usr/bin/env python3
"""Run the AAEC formal coefficient selector on a validation-sweep CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aaec.selection import annotate_and_select


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with alpha, val_bicubic_acc, val_bilinear_acc")
    parser.add_argument("--epsilon-pp", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    annotated, selected = annotate_and_select(
        read_rows(Path(args.input)),
        epsilon_pp=args.epsilon_pp,
    )
    write_rows(Path(args.output), annotated)
    print(json.dumps({
        "selected_alpha": float(selected["selected_alpha"]),
        "selection_rule": selected["selection_rule"],
        "selection_reference": selected["selection_reference"],
        "fallback_alpha1": bool(selected["selection_fallback_alpha1"]),
        "epsilon_pp": float(args.epsilon_pp),
    }, indent=2))


if __name__ == "__main__":
    main()
