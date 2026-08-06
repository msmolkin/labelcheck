"""Reading the application-record CSV in, and writing the results CSV out (§7.6).

The import is forgiving about *shape* and strict about *content*. An agent exporting from COLA
gets `TTB ID` or `Brand` and saves it from Excel with a byte-order mark and CRLF endings; none
of that is their mistake to fix. An unknown column is different: it means the file is not what
they think it is, so it is refused by name rather than silently ignored. The export carries the
machine verdict and any agent decision side by side, never one overwriting the other (§6.8).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from app.verdict import CHECK_IDS, Verdict

RECORD_COLUMNS = ("record_id", "brand_name", "class_type", "alcohol_content", "net_contents")
OPTIONAL_COLUMNS = ("filename",)  # explicit image pairing, when the agent has it (§7.7)

# Header spellings we accept, folded to lowercase with `_` and whitespace collapsed to a
# single space. Anything not listed here is an error, not a guess.
ALIASES = {
    alias: canonical
    for canonical, spellings in {
        "record_id": ("record id", "id", "ttb id", "cola id"),
        "brand_name": ("brand name", "brand"),
        "class_type": ("class type", "class/type", "class", "type"),
        "alcohol_content": ("alcohol content", "abv", "alcohol"),
        "net_contents": ("net contents", "volume", "size"),
        "filename": ("filename", "file", "image"),
    }.items()
    for alias in spellings
}
EXPORT_COLUMNS = (
    ("filename", "record_id", "pairing", "machine_verdict", "agent_decision")
    + CHECK_IDS
    + ("processing_ms", "low_confidence", "notes")
)
TEMPLATE_ROWS = (
    ("label_001", "Old Tom Distillery", "Straight Bourbon Whiskey", "45% Alc./Vol.", "750 mL"),
    ("label_002", "Stone's Throw", "London Dry Gin", "47% Alc./Vol. (94 Proof)", "750 mL"),
)


class CsvFormatError(ValueError):
    """The file cannot be read as an application-record CSV. The message is shown to the user."""


def _slug(header: str) -> str:
    return " ".join(header.replace("_", " ").split()).lower()


def _text(source: Path | bytes | str) -> str:
    """A `Path` is read from disk; `bytes` and `str` are the file's contents."""
    if isinstance(source, Path):
        source = source.read_bytes()
    if isinstance(source, bytes):
        source = source.decode("utf-8-sig", errors="replace")
    return source.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def read_records(source: Path | bytes | str) -> tuple[list[dict], list[str]]:
    """Parse the records CSV into canonical rows, plus warnings worth showing the agent.

    Raises `CsvFormatError` when the header cannot be understood. A row with no record id, or
    a repeat of one already seen, is skipped with a warning rather than failing the upload.
    """
    reader = csv.reader(io.StringIO(_text(source)))
    header = next(reader, None)
    if not header:
        raise CsvFormatError(
            "The file is empty. Expected a header row: " + ", ".join(RECORD_COLUMNS)
        )
    columns = [ALIASES.get(_slug(name)) for name in header]
    unknown = [n.strip() for n, c in zip(header, columns, strict=True) if c is None and n.strip()]
    if unknown:
        raise CsvFormatError(
            f"This CSV has column(s) we do not recognise: {', '.join(unknown)}. "
            f"Expected: {', '.join(RECORD_COLUMNS + OPTIONAL_COLUMNS)}."
        )
    if "record_id" not in columns:
        raise CsvFormatError(
            "This CSV has no record id column. Expected one of: record_id, id, ttb_id, cola_id."
        )
    rows: list[dict] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for line, values in enumerate(reader, start=2):
        if not any(value.strip() for value in values):
            continue
        row = dict.fromkeys(RECORD_COLUMNS + OPTIONAL_COLUMNS, "")
        for column, value in zip(columns, values, strict=False):
            if column:
                row[column] = value.strip()
        if not row["record_id"]:
            warnings.append(f"Row {line} has no record id and was skipped.")
        elif row["record_id"] in seen:
            warnings.append(f"Row {line} repeats record id {row['record_id']}; the first was kept.")
        else:
            seen.add(row["record_id"])
            rows.append(row)
    return rows, warnings


def template_csv() -> str:
    """The blank records CSV offered next to the upload control."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(RECORD_COLUMNS)
    writer.writerows(TEMPLATE_ROWS)
    return out.getvalue()


def result_row(meta: dict, verdict: Verdict) -> dict[str, str]:
    """One export row: the per-check states plus the reason codes behind them."""
    states = {check.id: check.state for check in verdict.checks}
    notes = [
        check.evidence["note_code"] for check in verdict.checks if check.evidence.get("note_code")
    ]
    if meta.get("low_confidence"):
        notes.append("low_confidence")
    row = {
        "filename": meta.get("filename", ""),
        "record_id": meta.get("record_id") or "",
        "pairing": meta.get("pairing") or "none",
        "machine_verdict": verdict.overall,
        "agent_decision": meta.get("agent_decision") or "",
        "processing_ms": str(meta.get("processing_ms") or ""),
        "low_confidence": "yes" if meta.get("low_confidence") else "no",
        "notes": ";".join(notes),
    }
    for check_id in CHECK_IDS:
        state = states.get(check_id, "na")
        row[check_id] = "n/a" if state == "na" else state
    return row


def write_results(items: Iterable[tuple[dict, Verdict]], stream: TextIO) -> None:
    """Write the §7.6 export. Column order is fixed: it is what the spreadsheet expects."""
    writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for meta, verdict in items:
        writer.writerow(result_row(meta, verdict))
