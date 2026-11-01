"""Explicit-output launcher for the versioned EHR-AGE analysis source package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import _runtime


def main():
    package = Path(__file__).resolve().parent
    metadata = json.loads((package / "package.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--task", choices=metadata["tasks"])
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-map", type=Path, help="JSON mapping of logical input paths to accessible files/directories")
    args, remaining = parser.parse_known_args()
    if args.list:
        print(json.dumps(metadata["tasks"], indent=2))
        return
    if args.task is None or args.project_root is None or args.output is None:
        parser.error("--task, --project-root and --output are required")
    project, output = args.project_root.resolve(), args.output.resolve()
    if not output.is_relative_to(project):
        parser.error("--output must be inside the explicitly selected project root")
    if output == package or output.is_relative_to(package):
        parser.error("Run outputs must be separate from the source package")
    declared = os.environ.get("PROJECT_RUN_DIR")
    if declared is not None and Path(declared).resolve() != output:
        parser.error("--output differs from the guarded run directory")
    os.environ["PROJECT_RUN_DIR"] = str(output)
    task = metadata["tasks"][args.task]
    output.mkdir(parents=True, exist_ok=True)
    input_map = json.loads(args.input_map.read_text(encoding="utf-8")) if args.input_map else {}
    _runtime.activate(output if task.get("root_mode") == "output" else project,
                      input_map=input_map, audit_project=project)
    os.environ.setdefault("NUMBA_CACHE_DIR", os.environ.get("PROJECT_CACHE_DIR", str(output / "cache")))
    _runtime.OVERRIDES.update({key: output / value for key, value in task.get("output_globals", {}).items()})
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    output_flag = task.get("output_flag")
    if output_flag and output_flag not in remaining:
        remaining += [output_flag, str(output)]
    if task.get("project_flag") and task["project_flag"] not in remaining:
        remaining += [task["project_flag"], str(project)]
    sys.argv = [str(_runtime.SOURCE / task["entry"]), *remaining]
    module = _runtime.load(task["entry"])
    if task.get("dispatch") == "parsed_run":
        result = module.run(module.parse_args())
    elif task.get("dispatch") == "parsed_kwargs":
        parsed = module.parse_args()
        result = getattr(module, task["function"])(**{key: getattr(parsed, key) for key in task["arguments"]})
    else:
        result = getattr(module, task.get("function", "main"))()
    if isinstance(result, int) and result:
        raise SystemExit(result)
    _runtime.assert_source_isolation()
    receipt = {"status": "COMPLETE", "task": args.task, "entry": task["entry"],
               "project_root": str(project), "output": str(output), "argv": remaining,
               "logical_project_root": str(_runtime.PROJECT), "input_map": input_map,
               "packaged_modules": sorted(_runtime.LOADED),
               "loaded_source_sha256": {p: hashlib.sha256((_runtime.SOURCE / p).read_bytes()).hexdigest() for p in _runtime.LOADED},
               "adapter_sha256": hashlib.sha256((package / "_runtime.py").read_bytes()).hexdigest(),
               "package_metadata_sha256": hashlib.sha256((package / "package.json").read_bytes()).hexdigest(),
               "python": sys.version}
    (output / ("execution_" + args.task + ".json")).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "task": args.task, "packaged_modules": len(_runtime.LOADED)}))


if __name__ == "__main__":
    main()
