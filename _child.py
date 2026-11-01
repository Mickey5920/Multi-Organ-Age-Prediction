"""Propagate the explicit project/source bindings to a study-script child."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import _runtime


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--entry", required=True)
    args, remaining = parser.parse_known_args()
    context = json.loads(os.environ["EHR_REPRO_CONTEXT"])
    _runtime.activate(context["project"], context["input_map"], context["audit_project"])
    sys.argv = [str(_runtime.SOURCE / args.entry), *remaining]
    code = 0
    try:
        _runtime.load(args.entry, name="__main__")
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code
    _runtime.assert_source_isolation()
    print(json.dumps({"event": "packaged_child", "entry": args.entry,
                      "loaded_packaged_modules": len(_runtime.LOADED), "exit_code": code}), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
