"""Export leads from SQLite to the formats a sales operator actually uses.

Export is a pure read: no scoring, no network, no LLM. What is in the database
is exactly what lands in the file, so a rerun or a manual edit cannot change the
numbers between the report and the spreadsheet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ORDERED_COLUMNS = [
    "name", "phone", "city", "state", "zip", "address", "category",
    "rating", "reviews", "tier", "fit", "has_website", "status",
    "qualifier_reason", "critic_status", "critic_notes",
    "website_url", "website_confidence", "website_method",
    "source", "source_url", "run_id",
]

RENAME = {
    "name": "Name",
    "phone": "Phone",
    "city": "City",
    "state": "State",
    "zip": "ZIP Code",
    "address": "Address",
    "category": "Category",
    "rating": "Rating",
    "reviews": "Reviews Count",
    "tier": "Tier",
    "fit": "Fit Score",
    "has_website": "Has Website",
    "status": "Status",
    "qualifier_reason": "Qualification Reason",
    "critic_status": "Critic Status",
    "critic_notes": "Critic Notes",
    "website_url": "Website URL",
    "website_confidence": "Website Confidence",
    "website_method": "Verdict Method",
    "source": "Source",
    "source_url": "Source URL",
    "run_id": "Run ID",
}


def to_dataframe(leads: Iterable[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    rows = list(leads)
    if not rows:
        cols = columns or ORDERED_COLUMNS
        return pd.DataFrame(columns=[RENAME.get(c, c) for c in cols])

    present = set(rows[0].keys())
    cols = [c for c in (columns or ORDERED_COLUMNS) if c in present]
    extra = [c for c in rows[0] if c not in cols and c != "candidate_id"]
    df = pd.DataFrame(rows)[cols + extra]
    df = df.rename(columns={**RENAME, **{c: c.replace("_", " ").title() for c in extra}})
    if "Has Website" in df.columns:
        df["Has Website"] = df["Has Website"].map({1: "yes", 0: "no"}).where(
            df["Has Website"].notna(), "unknown"
        )
    return df


def write(
    leads: Iterable[dict[str, Any]],
    path: Path,
    *,
    fmt: str,
    summary: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = to_dataframe(leads)

    fmt = fmt.lower().lstrip(".")
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif fmt == "jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in leads:
                fh.write(json.dumps(row, default=str) + "\n")
    elif fmt in ("xlsx", "excel"):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:  # type: ignore[arg-type]
            df.to_excel(writer, sheet_name="Leads", index=False)
            if summary:
                summary_df = pd.DataFrame(
                    [{"Metric": k, "Value": v} for k, v in summary.items()]
                )
                summary_df.to_excel(writer, sheet_name="Run Summary", index=False)
                # A little formatting so the file opens ready to use.
                ws = writer.sheets["Leads"]
                ws.freeze_panes = "A2"
                widths = {
                    "A": 34, "B": 17, "C": 16, "D": 7, "E": 10, "F": 40,
                    "G": 22, "H": 8, "I": 14, "J": 7, "K": 10,
                }
                for col, width in widths.items():
                    ws.column_dimensions[col].width = width
    else:
        raise ValueError(f"unsupported export format: {fmt}")

    return path


def export_all(
    leads: list[dict[str, Any]],
    directory: Path,
    *,
    state: str,
    formats: Iterable[str] = ("xlsx", "csv"),
    summary: dict[str, Any] | None = None,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        ext = "xlsx" if fmt in ("xlsx", "excel") else fmt
        path = directory / f"leads_{state.lower()}.{ext}"
        written.append(write(leads, path, fmt=fmt, summary=summary))
    return written
