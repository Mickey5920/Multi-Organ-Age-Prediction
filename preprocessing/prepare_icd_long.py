"""Prepare or read-only verify ICD long rows using participant + original Array.

Standard library only. build writes a NEW destination; verify never writes data.
Packed diagnosis lists require an explicit declaration that list position is the
original Array index. No missing code/date is removed before joining by index.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date
from functools import lru_cache
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ID = "Participant ID"
CODE = "Diagnoses - ICD10"
DATE = "Date of first in-patient diagnosis - ICD10"
OUTPUT = [ID, "ICD", "Disease_Description_Feature", "record_time", "source_array_index"]
PACKAGE = Path(__file__).resolve().parents[1]


class InputError(ValueError):
    """Invalid input; error messages do not disclose participant values."""


@lru_cache(maxsize=50000)
def normalize_date(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    # The audited exports use ISO dates or unambiguous year/month/day dates.
    match = re.fullmatch(r"(\d{4})([-/])(\d{1,2})\2(\d{1,2})(?: 00:00:00)?", value)
    if not match:
        raise InputError("Expected a date in YYYY-MM-DD or YYYY/M/D format")
    try:
        return date(int(match[1]), int(match[3]), int(match[4])).isoformat()
    except ValueError:
        raise InputError("Invalid calendar date") from None


def array_columns(headers, prefix):
    pattern = re.compile(re.escape(prefix) + r"(?: \| Instance (\d+))? \| Array (\d+)$")
    columns = {}
    for header in headers:
        match = pattern.fullmatch(header)
        if not match:
            continue
        if match[1] not in (None, "0"):
            raise InputError("This entry accepts singular/Instance 0 fields only")
        index = int(match[2])
        if index in columns:
            raise InputError("Duplicate logical Array index in input headers")
        columns[index] = header
    return dict(sorted(columns.items()))


def table(path):
    """Yield header then rows without converting opaque participant IDs."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            headers = next(reader)
        except StopIteration:
            raise InputError("Empty CSV input") from None
        if len(headers) != len(set(headers)) or ID not in headers:
            raise InputError("CSV requires unique headers and Participant ID")
        yield headers
        for number, values in enumerate(reader, 2):
            if len(values) != len(headers):
                raise InputError(f"CSV field-count mismatch at row {number}")
            row = dict(zip(headers, values))
            if not row[ID] or row[ID] != row[ID].strip():
                raise InputError(f"Missing or whitespace-padded participant key at row {number}")
            yield row


def parse_diagnosis(cell):
    cell = cell.strip()
    if not cell:
        return "", ""
    parts = cell.split(maxsplit=1)
    code = parts[0].strip("'")
    if not re.fullmatch(r"[A-Za-z][0-9][A-Za-z0-9.]*", code):
        raise InputError("Diagnosis cell must begin with an ICD-10 code")
    # Historical long-table preparation strips surrounding single quotes.
    description = (parts[1] if len(parts) == 2 else "").strip().strip("'")
    return code, description


def pair_arrays(codes, dates, allowed_date_arrays):
    """Return (original Array, code, description, date); never compact slots."""
    if any(index not in allowed_date_arrays for index in codes):
        raise InputError("Diagnosis Array has no corresponding date column")
    if any(value and not codes.get(index, "").strip() for index, value in dates.items()):
        raise InputError("Nonempty date has no diagnosis at the same Array index")
    records = []
    for index, cell in sorted(codes.items()):
        code, description = parse_diagnosis(cell)
        if code:
            records.append((index, code, description, dates.get(index, "")))
    # Match the study's one empty placeholder row for a diagnosis-free participant.
    return records or [(None, "", "", "")]


def progress(args, phase, participants):
    if args.progress_every and participants % args.progress_every == 0:
        print(json.dumps({"phase": phase, "participants": participants}), flush=True)


def iter_participants(args, counts):
    date_rows = table(args.dates_csv)
    date_headers = next(date_rows)
    date_columns = array_columns(date_headers, DATE)
    if not date_columns:
        raise InputError("No indexed diagnosis-date columns found")
    dates_by_id = {}
    for number, row in enumerate(date_rows, 2):
        pid = row[ID]
        if pid in dates_by_id:
            raise InputError(f"Duplicate participant key in date CSV at row {number}")
        # Missing values are omitted from this sparse lookup, while their original
        # indices remain in date_columns. A missing paired date is still blank.
        dates_by_id[pid] = {i: value for i, h in date_columns.items() if (value := normalize_date(row[h]))}
        progress(args, "read_dates", len(dates_by_id))
    counts["date_participants"] = len(dates_by_id)
    counts["date_array_columns"] = len(date_columns)
    code_rows = table(args.codes_csv)
    code_headers = next(code_rows)
    code_columns = array_columns(code_headers, CODE)
    shared_dates = array_columns(code_headers, DATE)
    if args.code_format == "packed":
        if not args.packed_order_is_array:
            raise InputError("Packed input requires --packed-order-is-array after verifying export order")
        if CODE not in code_headers or code_columns:
            raise InputError("Packed input requires a single Diagnoses - ICD10 field and no indexed code fields")
    elif not code_columns or set(code_columns) != set(date_columns):
        raise InputError("Indexed code/date headers must contain identical Array index sets")
    seen = set()
    for number, row in enumerate(code_rows, 2):
        pid = row[ID]
        if pid in seen:
            raise InputError(f"Duplicate participant key in code CSV at row {number}")
        seen.add(pid)
        if pid not in dates_by_id:
            raise InputError(f"Code participant has no date-table key at row {number}")
        dates = dates_by_id[pid]
        for index, header in shared_dates.items():
            if index not in date_columns or normalize_date(row[header]) != dates.get(index, ""):
                raise InputError(f"Shared original date Array differs at code CSV row {number}")
        if args.code_format == "packed":
            # Keep empty positions, including interior ones. Trailing empty cells
            # have no clinical content and are not invented as extra Array keys.
            codes = dict(enumerate(row[CODE].split("|"))) if row[CODE].strip() else {}
            while codes and not codes[max(codes)].strip():
                del codes[max(codes)]
        else:
            codes = {i: row[h] for i, h in code_columns.items()}
        records = pair_arrays(codes, dates, set(date_columns))
        counts["participants"] += 1
        counts["rows"] += len(records)
        counts["coded_rows"] += sum(bool(r[1]) for r in records)
        counts["empty_code_rows"] += sum(not r[1] for r in records)
        counts["undated_coded_rows"] += sum(bool(r[1]) and not r[3] for r in records)
        counts["shared_date_cells_checked"] += len(shared_dates)
        progress(args, "pair_arrays", counts["participants"])
        yield pid, records
    if seen != dates_by_id.keys():
        raise InputError("Date table contains participants absent from the code table")


def digest_rows(rows):
    # Sorting retains duplicate multiplicity; no delimited-string ambiguity.
    return hashlib.sha256(json.dumps(sorted(rows), ensure_ascii=False, separators=(",", ":")).encode()).digest()


def verify(args):
    existing = table(args.existing_long)
    headers = next(existing)
    if not set(OUTPUT[:4]) <= set(headers):
        raise InputError("Existing long table lacks the four required clinical columns")
    has_array = OUTPUT[4] in headers
    expected, counts = {}, Counter()
    for pid, records in iter_participants(args, counts):
        values = [(r[1], r[2], r[3], "" if r[0] is None else str(r[0])) for r in records]
        expected[pid] = digest_rows(values if has_array else [r[:3] for r in values])
    current, seen = Counter(), set()
    for pid, group in groupby(existing, key=lambda row: row[ID]):
        if pid in seen:
            raise InputError("Existing long table must keep each participant's rows contiguous")
        seen.add(pid)
        values = []
        for row in group:
            value = (row["ICD"], row["Disease_Description_Feature"], normalize_date(row["record_time"]))
            values.append(value + (row[OUTPUT[4]],) if has_array else value)
            current["rows"] += 1
            current["coded_rows"] += bool(row["ICD"])
            current["empty_code_rows"] += not row["ICD"]
            current["undated_coded_rows"] += bool(row["ICD"]) and not value[2]
        current["participants"] += 1
        if expected.get(pid) != digest_rows(values):
            current["mismatched_participants"] += 1
        progress(args, "compare_existing", current["participants"])
    current["missing_participants"] = len(expected.keys() - seen)
    current["unexpected_participants"] = len(seen - expected.keys())
    current.setdefault("mismatched_participants", 0)
    passed = (not current["mismatched_participants"] and not current["missing_participants"]
              and not current["unexpected_participants"] and current["rows"] == counts["rows"])
    return {"status": "PASS" if passed else "FAIL", "mode": "read_only_verify",
            "source": dict(counts), "existing": dict(current),
            "compared_fields": OUTPUT[:4] + ([OUTPUT[4]] if has_array else []),
            "comparison": "per-participant SHA256 of sorted row multisets; duplicate counts retained",
            "participant_data_written": False}


def build(args):
    output = args.output.resolve()
    if output.is_relative_to(PACKAGE):
        raise InputError("Output must be outside the source package")
    if output.exists() or output in {args.codes_csv.resolve(), args.dates_csv.resolve()}:
        raise InputError("Output must be a new file, distinct from all inputs")
    if not output.parent.is_dir():
        raise InputError("Create the intended output directory explicitly before building")
    counts = Counter()
    # Validate the complete input while writing a temp file on the selected
    # output filesystem. Open the final file exclusively only after validation;
    # a pre-existing destination, including one created concurrently, is preserved.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                                     prefix=".icd-array-", suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(OUTPUT)
            for pid, records in iter_participants(args, counts):
                for index, code, description, value in records:
                    writer.writerow([pid, code, description, value, "" if index is None else index])
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    created = False
    try:
        with output.open("xb") as destination:
            created = True
            with temporary.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        if created:
            output.unlink()
        raise
    finally:
        temporary.unlink()
    return {"status": "PASS", "mode": "build_new_file", "source": dict(counts),
            "columns": OUTPUT, "existing_files_overwritten": False}


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "verify"])
    parser.add_argument("--codes-csv", type=Path, required=True)
    parser.add_argument("--dates-csv", type=Path, required=True)
    parser.add_argument("--code-format", choices=["indexed", "packed"], default="indexed")
    parser.add_argument("--packed-order-is-array", action="store_true")
    parser.add_argument("--existing-long", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=100000, help="0 suppresses progress; only aggregate counts are printed")
    parser.add_argument("--expected-existing-sha256", help="Optional frozen-input identity gate for verify")
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    if args.mode == "verify" and (not args.existing_long or args.output):
        parser.error("verify requires --existing-long and does not accept --output")
    if args.mode == "build" and (not args.output or args.existing_long or args.expected_existing_sha256):
        parser.error("build requires --output and does not accept verification-only arguments")
    try:
        hashes = {"codes_csv": fingerprint(args.codes_csv), "dates_csv": fingerprint(args.dates_csv)}
        if args.mode == "verify":
            hashes["existing_long"] = fingerprint(args.existing_long)
            if args.expected_existing_sha256 and hashes["existing_long"] != args.expected_existing_sha256.lower():
                raise InputError("Existing long-table SHA256 differs from the supplied frozen identity")
        result = verify(args) if args.mode == "verify" else build(args)
        result["input_sha256"] = hashes
        result["scientific_fits_run"] = 0
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "PASS" else 1
    except (InputError, OSError, csv.Error) as error:
        print(json.dumps({"status": "ERROR", "error": str(error), "scientific_fits_run": 0}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
