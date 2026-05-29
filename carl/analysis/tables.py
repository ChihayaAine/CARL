"""Table generation.

Produces ``runs/<task>/tables/main.csv`` (and ``main.md``) by aggregating
per-policy Summary records.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import Summary


def write_main_table(
    summaries: Sequence[Summary],
    out_dir: Path | str,
    *,
    name: str = "main",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Util", "Res.", "CHR", "N"])
        for s in summaries:
            w.writerow([
                s.name, f"{s.util:.4f}", f"{s.res:.4f}",
                "N/A" if s.chr is None else f"{s.chr:.4f}", s.n,
            ])

    with open(md_path, "w") as f:
        f.write("| Method | Util | Res. | CHR | N |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for s in summaries:
            chr_str = "N/A" if s.chr is None else f"{s.chr:.3f}"
            f.write(f"| {s.name} | {s.util:.3f} | {s.res:.3f} | {chr_str} | {s.n} |\n")
    return csv_path


def write_summary_jsonl(records: Iterable[dict], out_path: Path | str) -> Path:
    import json
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path
