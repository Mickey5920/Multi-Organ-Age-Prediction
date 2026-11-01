from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from fixed_replication_runner import (
    _load_complete_receipt,
    aggregate_fivefold,
    DEFAULT_AGE_TARGETS,
    DEFAULT_D07_003_CONTRACT,
    DEFAULT_FOLD_MANIFEST,
    DEFAULT_OUTCOMES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PARTICIPANT_MASKS,
    DEFAULT_REPLICATION_CONTRACT,
    DEFAULT_REPLICATION_ROOT,
    DEFAULT_ROLE_ROOT,
    DEFAULT_FOLD_ROLES,
    FORMAL_FOLDS,
    build_role_artifacts,
    load_replication_contract,
    load_receipt_selection,
    load_role_registry,
    prepare_fold_evaluation,
    run_fold0_reuse,
    score_fixed_outer,
    sha256_file,
    validate_code_lineage,
    validate_data_lineage,
    validate_published_role_artifacts,
    write_receipt_selection,
)


TRAIN_WORKER = Path(__file__).resolve().with_name("fixed_replication_train_worker.py")


def _sha(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("expected an explicit 64-hex SHA256")
    return normalized


def _attempt(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is None:
        raise argparse.ArgumentTypeError("attempt ID must be 1-64 safe characters")
    return value


def _contract(args: argparse.Namespace):
    if args.contract.resolve() != DEFAULT_REPLICATION_CONTRACT.resolve():
        raise SystemExit("--contract must be the canonical fixed-replication contract")
    return load_replication_contract(
        DEFAULT_REPLICATION_CONTRACT,
        expected_sha256=args.expected_contract_sha256,
    )


def _validate(args: argparse.Namespace) -> dict[str, object]:
    if not args.expected_role_registry_sha256 or not args.expected_role_validation_sha256:
        raise SystemExit("validate-only requires the frozen registry and role-validation SHA256 values")
    contract = _contract(args)
    code = validate_code_lineage(contract)
    data = validate_data_lineage(contract)
    result: dict[str, object] = {
        "status": "VALID",
        "replication_contract_sha256": contract.sha256,
        "code_lineage": code,
        "data_lineage": data,
        "formal_training_started": False,
    }
    if getattr(args, "expected_role_registry_sha256", None):
        registry = load_role_registry(
            DEFAULT_ROLE_ROOT / "ROLE_ARTIFACT_REGISTRY.FROZEN.json",
            contract,
            expected_sha256=args.expected_role_registry_sha256,
        )
        result["role_registry_folds"] = len(registry["folds"])
        report = validate_published_role_artifacts(
            contract=contract,
            role_root=DEFAULT_ROLE_ROOT,
            expected_registry_sha256=args.expected_role_registry_sha256,
            expected_validation_sha256=args.expected_role_validation_sha256,
        )
        result["role_artifact_validation"] = report
    return result


def _run_fold(args: argparse.Namespace, fold: int) -> Path:
    contract = _contract(args)
    validate_code_lineage(contract)
    validate_published_role_artifacts(
        contract=contract,
        role_root=DEFAULT_ROLE_ROOT,
        expected_registry_sha256=args.expected_role_registry_sha256,
        expected_validation_sha256=args.expected_role_validation_sha256,
    )
    if fold == 0:
        output = DEFAULT_OUTPUT_ROOT / "outer_fold_0" / "reuse_d07_003_e55"
        if output.exists():
            receipt = output / "FOLD_COMPLETE_RECEIPT.json"
            if receipt.is_file():
                _load_complete_receipt(receipt, contract, 0)
                return receipt
            raise FileExistsError("incomplete Fold0 reuse directory must be preserved")
        return run_fold0_reuse(
            contract_path=DEFAULT_REPLICATION_CONTRACT,
            expected_contract_sha256=args.expected_contract_sha256,
            role_registry_path=DEFAULT_ROLE_ROOT / "ROLE_ARTIFACT_REGISTRY.FROZEN.json",
            expected_role_registry_sha256=args.expected_role_registry_sha256,
            expected_role_validation_sha256=args.expected_role_validation_sha256,
            output_dir=output,
        )
    if fold not in FORMAL_FOLDS:
        raise ValueError("outer fold must be one of 0..4")
    registry_path = DEFAULT_ROLE_ROOT / "ROLE_ARTIFACT_REGISTRY.FROZEN.json"
    registry = load_role_registry(
        registry_path,
        contract,
        expected_sha256=args.expected_role_registry_sha256,
    )
    row = next(item for item in registry["folds"] if int(item["outer_fold"]) == fold)
    if row.get("mode") != "train_from_scratch":
        raise RuntimeError(f"outer fold {fold} registry mode changed")
    authorization_path = Path(row["train_authorization_path"]).resolve()
    authorization_sha = str(row["train_authorization_sha256"])
    attempt_dir = DEFAULT_OUTPUT_ROOT / f"outer_fold_{fold}" / "attempts" / args.attempt_id
    training_complete = attempt_dir / "TRAINING_COMPLETE.json"
    if attempt_dir.exists():
        canonical_receipt = attempt_dir / "score" / "FOLD_COMPLETE_RECEIPT.json"
        if canonical_receipt.is_file():
            _load_complete_receipt(canonical_receipt, contract, fold)
            return canonical_receipt
        explicit_receipt = getattr(args, "completed_receipt", None)
        if explicit_receipt is not None:
            explicit_receipt = explicit_receipt.resolve()
            if attempt_dir not in explicit_receipt.parents:
                raise SystemExit("--completed-receipt must belong to the explicit attempt directory")
            if not args.expected_completed_receipt_sha256 or sha256_file(explicit_receipt) != args.expected_completed_receipt_sha256:
                raise SystemExit("explicit completed receipt SHA256 mismatch")
            _load_complete_receipt(explicit_receipt, contract, fold)
            return explicit_receipt
        if not training_complete.is_file():
            raise FileExistsError(
                f"training attempt is incomplete and cannot resume; preserve it and restart E1 in a new attempt: {attempt_dir}"
            )
        resume_id = getattr(args, "evaluation_resume_id", None)
        if resume_id is None:
            raise SystemExit(
                "E60 is sealed; pass an explicit --evaluation-resume-id to continue without retraining"
            )
        evaluation_dir = attempt_dir / f"evaluation_resume_{resume_id}"
        score_dir = attempt_dir / f"score_resume_{resume_id}"
    else:
        command = [
            sys.executable,
            str(TRAIN_WORKER),
            "--train-authorization",
            str(authorization_path),
            "--expected-train-authorization-sha256",
            authorization_sha,
            "--expected-contract-sha256",
            args.expected_contract_sha256,
            "--expected-role-validation-sha256",
            args.expected_role_validation_sha256,
            "--attempt-dir",
            str(attempt_dir),
        ]
        subprocess.run(command, check=True)
        evaluation_dir = attempt_dir / "evaluation"
        score_dir = attempt_dir / "score"
    training_sha = sha256_file(training_complete)
    evaluation_manifest, ledger = prepare_fold_evaluation(
        contract_path=DEFAULT_REPLICATION_CONTRACT,
        expected_contract_sha256=args.expected_contract_sha256,
        role_registry_path=registry_path,
        expected_role_registry_sha256=args.expected_role_registry_sha256,
        expected_role_validation_sha256=args.expected_role_validation_sha256,
        training_complete_path=training_complete,
        expected_training_complete_sha256=training_sha,
        evaluation_dir=evaluation_dir,
    )
    return score_fixed_outer(
        contract_path=DEFAULT_REPLICATION_CONTRACT,
        expected_contract_sha256=args.expected_contract_sha256,
        role_registry_path=registry_path,
        expected_role_registry_sha256=args.expected_role_registry_sha256,
        expected_role_validation_sha256=args.expected_role_validation_sha256,
        training_complete_path=training_complete,
        expected_training_complete_sha256=training_sha,
        evaluation_manifest_path=evaluation_manifest,
        expected_evaluation_manifest_sha256=sha256_file(evaluation_manifest),
        ledger_path=ledger,
        expected_ledger_sha256=sha256_file(ledger),
        score_dir=score_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D07-003 fixed E55 five-fold replication")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "build-role-artifacts",
        "validate-only",
        "run-fold",
        "run-sequence",
        "select-receipts",
        "aggregate-5fold",
    ):
        child = subparsers.add_parser(name)
        child.add_argument("--contract", type=Path, default=DEFAULT_REPLICATION_CONTRACT)
        child.add_argument("--expected-contract-sha256", type=_sha, required=True)
        if name in {"validate-only", "run-fold", "run-sequence", "select-receipts", "aggregate-5fold"}:
            child.add_argument("--expected-role-registry-sha256", type=_sha)
            child.add_argument("--expected-role-validation-sha256", type=_sha)
        if name in {"run-fold", "run-sequence"}:
            child.add_argument("--attempt-id", type=_attempt, required=True)
        if name == "run-fold":
            child.add_argument("--fold", type=int, choices=range(5), required=True)
            child.add_argument("--completed-receipt", type=Path)
            child.add_argument("--expected-completed-receipt-sha256", type=_sha)
            child.add_argument("--evaluation-resume-id", type=_attempt)
        if name == "run-sequence":
            child.add_argument("--outer-folds", default="0,1,2,3,4")
        if name == "aggregate-5fold":
            child.add_argument("--receipt-selection", type=Path, required=True)
            child.add_argument("--expected-receipt-selection-sha256", type=_sha, required=True)
        if name == "select-receipts":
            child.add_argument("--selection-id", type=_attempt, required=True)
            child.add_argument(
                "--receipt",
                action="append",
                required=True,
                help="Explicit outer=path@sha256 binding; pass exactly five times.",
            )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _contract(args)
    if args.command == "validate-only":
        print(json.dumps(_validate(args), indent=2), flush=True)
        return
    if args.command == "build-role-artifacts":
        result = build_role_artifacts(
            contract_path=DEFAULT_REPLICATION_CONTRACT,
            expected_contract_sha256=args.expected_contract_sha256,
            outcomes_path=DEFAULT_OUTCOMES,
            age_targets_path=DEFAULT_AGE_TARGETS,
            fold_manifest_path=DEFAULT_FOLD_MANIFEST,
            outer_fold_roles_path=DEFAULT_FOLD_ROLES,
            participant_masks_path=DEFAULT_PARTICIPANT_MASKS,
            output_root=DEFAULT_ROLE_ROOT,
        )
        validation = DEFAULT_ROLE_ROOT / "ROLE_ARTIFACT_VALIDATION.PASS.json"
        print(
            json.dumps(
                {
                    "registry": str(result),
                    "registry_sha256": sha256_file(result),
                    "validation": str(validation),
                    "validation_sha256": sha256_file(validation),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if not args.expected_role_registry_sha256 or not args.expected_role_validation_sha256:
        raise SystemExit(
            "--expected-role-registry-sha256 and --expected-role-validation-sha256 are required"
        )
    if args.command == "run-fold":
        print(_run_fold(args, args.fold), flush=True)
        return
    if args.command == "run-sequence":
        folds = tuple(int(value) for value in args.outer_folds.split(","))
        if not folds or len(set(folds)) != len(folds) or not set(folds).issubset(set(range(5))):
            raise SystemExit("--outer-folds must be a unique comma-separated subset of 0..4")
        receipt_results = {fold: _run_fold(args, fold) for fold in folds}
        results = [str(receipt_results[fold]) for fold in folds]
        payload: dict[str, object] = {"receipts": results}
        if set(folds) == set(range(5)):
            contract = _contract(args)
            selection_root = DEFAULT_OUTPUT_ROOT / "receipt_selections"
            selection_root.mkdir(parents=True, exist_ok=True)
            selection_path = selection_root / f"{args.attempt_id}.json"
            selected = {fold: receipt_results[fold] for fold in range(5)}
            write_receipt_selection(
                path=selection_path,
                contract=contract,
                role_registry_sha256=args.expected_role_registry_sha256,
                receipt_paths=selected,
            )
            payload["receipt_selection"] = str(selection_path)
            payload["receipt_selection_sha256"] = sha256_file(selection_path)
        print(json.dumps(payload, indent=2), flush=True)
        return
    if args.command == "select-receipts":
        contract = _contract(args)
        selected: dict[int, Path] = {}
        for value in args.receipt:
            match = re.fullmatch(r"([0-4])=(.+)@([0-9a-fA-F]{64})", value)
            if match is None:
                raise SystemExit("each --receipt must be outer=path@sha256")
            outer = int(match.group(1))
            path = Path(match.group(2)).resolve()
            if outer in selected or DEFAULT_OUTPUT_ROOT.resolve() not in path.parents:
                raise SystemExit("receipt folds must be unique and paths must be under canonical runs")
            if sha256_file(path) != match.group(3).lower():
                raise SystemExit(f"receipt SHA256 mismatch for outer {outer}")
            selected[outer] = path
        selection_root = DEFAULT_OUTPUT_ROOT / "receipt_selections"
        selection_root.mkdir(parents=True, exist_ok=True)
        selection_path = selection_root / f"{args.selection_id}.json"
        write_receipt_selection(
            path=selection_path,
            contract=contract,
            role_registry_sha256=args.expected_role_registry_sha256,
            receipt_paths=selected,
        )
        print(
            json.dumps(
                {"receipt_selection": str(selection_path), "sha256": sha256_file(selection_path)},
                indent=2,
            ),
            flush=True,
        )
        return
    if args.command == "aggregate-5fold":
        contract = _contract(args)
        selection = args.receipt_selection.resolve()
        selection_root = (DEFAULT_OUTPUT_ROOT / "receipt_selections").resolve()
        if selection.parent != selection_root:
            raise SystemExit("--receipt-selection must be under the canonical receipt_selections root")
        receipts = load_receipt_selection(
            path=selection,
            expected_sha256=args.expected_receipt_selection_sha256,
            contract=contract,
            role_registry_sha256=args.expected_role_registry_sha256,
        )
        result = aggregate_fivefold(
            contract_path=DEFAULT_REPLICATION_CONTRACT,
            expected_contract_sha256=args.expected_contract_sha256,
            role_registry_path=DEFAULT_ROLE_ROOT / "ROLE_ARTIFACT_REGISTRY.FROZEN.json",
            expected_role_registry_sha256=args.expected_role_registry_sha256,
            expected_role_validation_sha256=args.expected_role_validation_sha256,
            receipt_paths=receipts,
            receipt_selection_path=selection,
            expected_receipt_selection_sha256=args.expected_receipt_selection_sha256,
        )
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
