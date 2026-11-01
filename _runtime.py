"""Path-only execution adapter for the byte-preserved study source snapshot.

Numeric code is not rewritten. The adapter changes author-project path discovery,
source imports and reads so the snapshot can use a separately supplied data root.
It never copies participant data and never silently imports an original .py file.
"""
from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import os
from pathlib import Path, PureWindowsPath
import sys
import subprocess

PACKAGE = Path(__file__).resolve().parent
SOURCE = Path("\\\\?\\" + str(PACKAGE / "source")) if os.name == "nt" else PACKAGE / "source"
PROJECT = SOURCE
AUDIT_PROJECT = SOURCE
INPUT_MAP = {}
ORIGINAL_PROJECT = PureWindowsPath(r"E:\MPU\Code\Age-Prediction-0620")
OVERRIDES = {}
LOADED = {}
_original_get_code = importlib.machinery.SourceFileLoader.get_code
_original_spec = importlib.util.spec_from_file_location
_original_popen = subprocess.Popen


def project_dir(relative=""):
    return PROJECT / relative


def _logical(path):
    if isinstance(path, int) or hasattr(path, "read"):
        return path
    value = os.fspath(path)
    # Logical paths may already lie in a nested reproduction workspace under
    # the original checkout. Reapplying the original-root substitution would
    # duplicate that workspace prefix and break packaged sibling imports.
    candidate = Path(value)
    if candidate.is_absolute() and any(candidate.is_relative_to(root) for root in (SOURCE, PROJECT)):
        return candidate
    try:
        relative = PureWindowsPath(value).relative_to(ORIGINAL_PROJECT)
    except ValueError:
        return Path(value)
    return PROJECT.joinpath(*relative.parts)


def _relative(path):
    for root in (SOURCE, PROJECT):
        try:
            return Path(path).relative_to(root)
        except ValueError:
            pass
    return None


def source_path(path):
    value = _logical(path)
    if not isinstance(value, Path):
        return value
    relative = _relative(value)
    if relative is None:
        return value
    packaged = SOURCE / relative
    if packaged.exists():
        return packaged
    if value.suffix == ".py":
        raise FileNotFoundError(f"Source dependency is not in this package: {relative}")
    return value


def input_path(path):
    value = _logical(path)
    if not isinstance(value, Path):
        return value
    key = _path_key(value)
    for old, new in sorted(INPUT_MAP.items(), key=lambda pair: len(str(pair[0])), reverse=True):
        if key == old or key.startswith(old.rstrip("/") + "/"):
            suffix = os.fspath(value).replace("\\", "/").rstrip("/")[len(old):].lstrip("/")
            return Path(new).joinpath(*suffix.split("/")) if suffix else Path(new)
    relative = _relative(value)
    if relative is None:
        return value
    packaged, data = SOURCE / relative, PROJECT / relative
    if value.suffix == ".py":
        return source_path(value)
    if data.exists():
        return data
    if packaged.exists():
        return packaged
    return data


def _path_key(path):
    """Compare Windows logical input paths on either host operating system."""
    value = os.fspath(path).replace("\\", "/").rstrip("/")
    return value.casefold() if PureWindowsPath(value).drive else value


def interpreter_path(path):
    """Explicit interpreter maps win; otherwise use this task's interpreter."""
    value = input_path(path)
    if value != Path(path):
        return str(value)
    if PureWindowsPath(os.fspath(path)).name.lower() in ("python.exe", "pythonw.exe"):
        return sys.executable
    return os.fspath(path)


def creation_no_window():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def prepare_run_root(path):
    root = Path(path).resolve()
    if not root.is_relative_to(PROJECT.resolve()):
        raise ValueError("Output must be within the explicitly selected project root")
    for name in ("outputs", "cache", "tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", os.environ.get("PROJECT_CACHE_DIR", str(root / "cache")))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", os.environ.get("TMP", str(root / "tmp")))
    return root


def _call(name, value):
    return ast.Call(ast.Attribute(ast.Name("_ehr_repro", ast.Load()), name, ast.Load()), [value], [])


def _file_ancestor(node, source_file):
    """Resolve only expressions based on this source file, never arbitrary code."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
        if len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == "__file__":
            return source_file
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute"):
        return _file_ancestor(node.func.value, source_file)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _file_ancestor(node.value, source_file)
        return None if base is None else base.parent
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        base = _file_ancestor(node.value.value, source_file)
        index = node.slice.value if isinstance(node.slice, ast.Constant) else None
        if base is not None and isinstance(index, int):
            return base.parents[index]
    return None


class PathAdapter(ast.NodeTransformer):
    def __init__(self, source_file):
        self.source_file = source_file

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "subprocess" and node.attr == "CREATE_NO_WINDOW":
            return ast.copy_location(ast.Call(ast.Attribute(ast.Name("_ehr_repro", ast.Load()), "creation_no_window", ast.Load()), [], []), node)
        ancestor = _file_ancestor(node, self.source_file)
        if ancestor is not None and ancestor.is_relative_to(SOURCE):
            return ast.copy_location(_call("project_dir", ast.Constant(ancestor.relative_to(SOURCE).as_posix())), node)
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        ancestor = _file_ancestor(node, self.source_file)
        if ancestor is not None and ancestor.is_relative_to(SOURCE):
            return ast.copy_location(_call("project_dir", ast.Constant(ancestor.relative_to(SOURCE).as_posix())), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        # Preserve an unmodified ancestor expression until it has been resolved.
        ancestor = _file_ancestor(node, self.source_file)
        if ancestor is not None and ancestor.is_relative_to(SOURCE):
            return ast.copy_location(_call("project_dir", ast.Constant(ancestor.relative_to(SOURCE).as_posix())), node)
        if isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            try:
                relative = PureWindowsPath(node.args[0].value).relative_to(ORIGINAL_PROJECT)
            except ValueError:
                pass
            else:
                return ast.copy_location(_call("project_dir", ast.Constant(relative.as_posix())), node)
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            name, owner = node.func.attr, node.func.value
            if name == "spec_from_file_location" and len(node.args) >= 2:
                node.args[1] = _call("source_path", node.args[1])
            elif name == "run_path" and isinstance(owner, ast.Name) and owner.id == "runpy" and node.args:
                node.func = ast.Attribute(ast.Name("_ehr_repro", ast.Load()), "run_source_path", ast.Load())
            elif (name in ("insert", "append", "extend") and isinstance(owner, ast.Attribute)
                  and isinstance(owner.value, ast.Name) and owner.value.id == "sys" and owner.attr == "path"):
                if name == "extend":
                    node.args[0] = ast.ListComp(ast.Call(ast.Name("str", ast.Load()), [_call("source_path", ast.Name("_p", ast.Load()))], []),
                        [ast.comprehension(ast.Name("_p", ast.Store()), node.args[0], [], 0)])
                else:
                    node.args[-1] = ast.Call(ast.Name("str", ast.Load()), [_call("source_path", node.args[-1])], [])
            elif name in ("read_text", "read_bytes", "exists", "is_file", "is_dir", "iterdir", "glob", "rglob", "stat"):
                node.func.value = _call("input_path", owner)
            elif name == "open":
                mode = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else "r"
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if isinstance(mode, str) and not any(x in mode for x in "wax+"):
                    node.func.value = _call("input_path", owner)
            elif name.startswith("read_") or name in ("load", "memmap", "loadtxt", "genfromtxt"):
                if node.args and isinstance(owner, ast.Name) and owner.id in ("pd", "pandas", "np", "numpy", "torch", "pl", "polars"):
                    node.args[0] = _call("input_path", node.args[0])
        elif isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
            if isinstance(mode, str) and not any(x in mode for x in "wax+"):
                node.args[0] = _call("input_path", node.args[0])
        return node


def adapted_tree(source_file):
    tree = ast.parse(source_file.read_text(encoding="utf-8-sig"), filename=str(source_file))
    tree = PathAdapter(source_file).visit(tree)
    insertion = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str) else 0
    while insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom) and tree.body[insertion].module == "__future__":
        insertion += 1
    tree.body.insert(insertion, ast.Import([ast.alias("_runtime", "_ehr_repro")]))
    # Replace the author-machine E:-drive gate with a project-bound output gate.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check_paths" and source_file.name == "run_information_matched.py":
            node.body = [ast.Return(_call("prepare_run_root", ast.Name(node.args.args[0].arg, ast.Load())))]
    return ast.fix_missing_locations(tree)


def activate(project, input_map=None, audit_project=None):
    global PROJECT, INPUT_MAP, AUDIT_PROJECT
    PROJECT = Path(project).resolve()
    AUDIT_PROJECT = Path(audit_project).resolve() if audit_project else PROJECT
    if not PROJECT.is_dir():
        raise FileNotFoundError(PROJECT)
    sys.dont_write_bytecode = True
    os.environ["EHR_AGE_PROJECT_ROOT"] = str(PROJECT)
    INPUT_MAP = {}
    for old, new in (input_map or {}).items():
        old = _logical(old)
        if not old.is_absolute() and not PureWindowsPath(str(old)).is_absolute():
            old = PROJECT / old
        INPUT_MAP[_path_key(old)] = Path(new).resolve()

    def get_code(loader, fullname):
        path = Path(loader.path).resolve()
        if not path.is_relative_to(SOURCE):
            return _original_get_code(loader, fullname)
        LOADED[path.relative_to(SOURCE).as_posix()] = True
        return compile(adapted_tree(path), str(path), "exec", dont_inherit=True)

    def spec(name, location=None, **kwargs):
        return _original_spec(name, None if location is None else source_path(location), **kwargs)

    importlib.machinery.SourceFileLoader.get_code = get_code
    importlib.util.spec_from_file_location = spec
    context = json_context()

    class ReproPopen(_original_popen):
        def __init__(self, args, *positional, **kwargs):
            if isinstance(args, (list, tuple)) and args:
                command = list(args)
                command[0] = interpreter_path(command[0])
                for index, item in enumerate(command[1:], 1):
                    if not isinstance(item, (str, Path)) or not str(item).endswith(".py"):
                        continue
                    path = _logical(item)
                    if not path.is_absolute():
                        path = Path(kwargs.get("cwd") or Path.cwd()) / path
                    path = path.resolve()
                    if path.is_relative_to(PACKAGE) and not path.is_relative_to(SOURCE):
                        continue
                    if _relative(path) is None:
                        continue
                    source = source_path(path)
                    relative = source.relative_to(SOURCE).as_posix()
                    command[index:index + 1] = [str(PACKAGE / "_child.py"), "--entry", relative]
                    env = dict(kwargs.get("env") or os.environ)
                    env["EHR_REPRO_CONTEXT"] = context
                    kwargs["env"] = env
                    args = command
                    break
            super().__init__(args, *positional, **kwargs)

    subprocess.Popen = ReproPopen


def json_context():
    import json
    return json.dumps({"project": str(PROJECT), "audit_project": str(AUDIT_PROJECT),
                       "input_map": {str(k): str(v) for k, v in INPUT_MAP.items()}})


def run_source_path(path, init_globals=None, run_name=None):
    import types
    source = source_path(path)
    name = run_name or "<run_path>"
    module = types.ModuleType(name)
    if init_globals:
        module.__dict__.update(init_globals)
    module.__dict__.update(__name__=name, __file__=str(source), __package__=None)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(adapted_tree(source), str(source), "exec", dont_inherit=True), module.__dict__)
        return dict(module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load(relative, name="ehr_repro_entry"):
    path = SOURCE / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for key, value in OVERRIDES.items():
        if hasattr(module, key):
            setattr(module, key, value)
    return module


def assert_source_isolation():
    unexpected = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or not str(file).endswith(".py"):
            continue
        path = Path(file).resolve()
        # torch.ops/classes are proxy modules with synthetic, non-existent
        # __file__ values; they are not imported project source files.
        if not path.is_file():
            continue
        if path.is_relative_to(AUDIT_PROJECT) and not path.is_relative_to(PACKAGE):
            unexpected.append((name, str(path)))
    if unexpected:
        raise RuntimeError(f"Original project Python code was imported instead of packaged code: {unexpected}")
