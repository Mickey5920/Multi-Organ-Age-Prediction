from __future__ import annotations

import argparse
import json

from v8_execution_guard import current_environment_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--packages", required=True)
    args = parser.parse_args()
    record = current_environment_record(
        role=args.role,
        required_packages=[value for value in args.packages.split(",") if value],
    )
    print(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
