from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "Revision-0710" / "01_data_rebuild" / "00_config" / "source_paths.json"
OUTPUT = ROOT / "Revision-0710" / "00_contract" / "SOURCE_FINGERPRINTS.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    configured = {
        key: value
        for key, value in config.items()
        if isinstance(value, str) and (key.endswith("_csv") or key.endswith("_script"))
    }
    fingerprints = {}
    for key, value in configured.items():
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Configured source does not exist: {key}={path}")
        stat = path.stat()
        print(f"Hashing {key}: {path} ({stat.st_size:,} bytes)", flush=True)
        fingerprints[key] = {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "sha256": sha256_file(path),
        }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SHA256",
        "config_sha256": sha256_file(CONFIG),
        "sources": fingerprints,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
