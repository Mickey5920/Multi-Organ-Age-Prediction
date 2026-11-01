"""Maintain explicit delivery hashes and verify Git byte identity; never pushes.

--refresh updates only already-listed delivery entries after checking retained
source hashes. Additions require an explicit manifest change; no directory is
automatically swept into the release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "package.json"


def local_path(relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts or ".git" in parts.parts:
        raise ValueError("Manifest path is outside the release tree")
    path = ROOT.joinpath(*parts.parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Manifest path resolves outside the release tree")
    return Path("\\\\?\\" + str(resolved)) if os.name == "nt" else resolved


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_id(meta):
    return hashlib.sha256(json.dumps({"version": meta["version"], "files": meta["files"],
                                     "delivery_files": meta["delivery_files"]}, sort_keys=True).encode()).hexdigest()


def check_sources(meta):
    for item in meta["files"]:
        path = local_path("source/" + item["path"])
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Retained study-source hash differs: " + item["path"])


def refresh(meta):
    check_sources(meta)
    for item in meta["delivery_files"]:
        path = local_path(item["path"])
        item.update(sha256=sha(path), bytes=path.stat().st_size)
    meta["package_id"] = package_id(meta)
    MANIFEST.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check(meta, check_index=False):
    check_sources(meta)
    paths = ["source/" + x["path"] for x in meta["files"]] + [x["path"] for x in meta["delivery_files"]]
    if len(paths) != len(set(paths)) or "package.json" in paths:
        raise ValueError("Duplicate/self-referential manifest path")
    for item in meta["delivery_files"]:
        path = local_path(item["path"])
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Delivery hash differs: " + item["path"])
    if package_id(meta) != meta["package_id"]:
        raise ValueError("Package identity differs from its catalog")
    if meta["version"] != meta["version_control"]["release_tag"]:
        raise ValueError("Source version and release-tag name differ")
    report = {"status": "PASS", "version": meta["version"], "package_id": meta["package_id"],
              "retained_source_files": len(meta["files"]), "delivery_files": len(meta["delivery_files"]),
              "git_index_checked": False, "push_or_upload_performed": False}
    if check_index:
        result = subprocess.run(["git", "-c", "core.longpaths=true", "ls-files", "--stage", "-z"],
                                cwd=ROOT, check=True, capture_output=True)
        tracked = {}
        for entry in result.stdout.split(b"\0"):
            if not entry:
                continue
            record, name = entry.split(b"\t", 1)
            mode, blob, stage = record.decode("ascii").split()
            if stage != "0" or mode not in ("100644", "100755"):
                raise ValueError("Unmerged or unsupported Git index entry")
            tracked[name.decode("utf-8")] = blob
        if set(tracked) != set(paths) | {"package.json"}:
            raise ValueError("Git index file set differs from the explicit release catalog")
        for name, blob in tracked.items():
            raw = local_path(name).read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if actual != blob:
                raise ValueError("Staged Git content differs from verified bytes: " + name)
        report.update(git_index_checked=True, git_index_files=len(tracked))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--check-index", action="store_true", help="Check the explicit staged file set and exact Git blob bytes too")
    mode.add_argument("--refresh", action="store_true", help="Refresh only the explicitly listed delivery files; retained source must match")
    args = parser.parse_args()
    try:
        meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if args.refresh:
            refresh(meta)
        print(json.dumps(check(meta, check_index=args.check_index)))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, "Release check failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
