"""One-shot schema-only149 integration check; never runs financial features."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch

import monthly_stock_comp149_runner as r

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = "work/monthly_stock_comp149_proposal.json"
PROPOSAL_SHA = "ae7a95712c4de477dbac3fa34daf46cdb2bf16074d504adb0a317f4b4bda3f5b"
OUTPUT = "work/monthly_stock_comp149_source_integration_check.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(mapping):
    r.require(type(mapping) is dict and bool(mapping), "Nonempty declared closure required")
    result = {}
    for name, sha in mapping.items():
        path = Path(name)
        absolute = path if path.is_absolute() else ROOT/path
        r.require(not any(s in str(absolute).lower() for s in ("onedrive", "cloudstorage")),
                  "Cloud path forbidden")
        r.require(type(sha) is str and re.fullmatch("[0-9a-f]{64}", sha), "SHA256 required")
        key = str(absolute.resolve())
        r.require(key not in result or result[key] == sha, "Conflicting closure path")
        result[key] = sha
    return result


def main():
    inputs = {}
    checked = []

    def read_ref(ref):
        r.require(type(ref) is dict and set(ref) == {"path", "sha256"}, "Exact input reference")
        path = Path(ref["path"])
        path = path if path.is_absolute() else ROOT/path
        r.require(path.resolve() == path and path.is_relative_to(ROOT),
                  "Actual input must be a local regular non-symlink file")
        raw = path.read_bytes()
        r.require(digest(raw) == ref["sha256"], "Input SHA mismatch: "+str(path))
        inputs[str(path)] = ref["sha256"]
        return json.loads(raw)

    for filename in (Path(__file__), Path(r.__file__), Path(r.factor.__file__),
                     Path(r.factor.old.__file__), Path(r.base.__file__)):
        inputs[str(filename.resolve())] = digest(filename.read_bytes())
    report = dict(
        schema_version=1, round=149, stage="source_schema_integration_check",
        created_utc=datetime.now(timezone.utc).isoformat(), passed=False,
        proposal_sha256=PROPOSAL_SHA, financial_feature_calculations=0,
        model_fits=0, score_calculations=0, rank_calculations=0,
        order_calculations=0, return_calculations=0, registry_writes=0,
        full_source_dependency_bytes_reaudited=False)
    try:
        proposal = read_ref(dict(path=PROPOSAL, sha256=PROPOSAL_SHA))
        refs = {key: proposal["sources"][key] for key in r.SBC_REFS}
        for key, ref in refs.items():
            r.require((ROOT/ref["path"]).resolve() == ROOT/r.SBC_REFS[key],
                      "Wrong final source package path: "+key)
        objects = {key: read_ref(ref) for key, ref in refs.items()}
        original = read_ref(proposal["sources"]["dataset"])
        original_receipt = read_ref(proposal["sources"]["admission"])
        # A regression invoking either builder must fail instead of silently
        # turning this metadata check into a real feature calculation.
        with patch.object(r.factor, "build_features", side_effect=AssertionError("Feature calculation forbidden")), \
             patch.object(r.factor.old, "build_features", side_effect=AssertionError("Original feature calculation forbidden")):
            r.validate_proposal(proposal)
            checked.append("validate_proposal")
            admitted, held = r.validate_sbc_admission(
                objects["sbc_dataset"], objects["sbc_admission"], objects["sbc_policy"],
                objects["sbc_review"], original, original_receipt, refs)
            checked.append("validate_sbc_admission")
        declared = normalized(objects["sbc_admission"]["input_hashes"])
        reviewed = normalized(objects["sbc_review"]["input_hashes"])
        needed = dict(declared)
        for key in ("sbc_dataset", "sbc_admission", "sbc_policy"):
            needed[str((ROOT/refs[key]["path"]).resolve())] = refs[key]["sha256"]
        r.require(all(reviewed.get(path) == sha for path, sha in needed.items()),
                  "Independent source review omitted a declared admission dependency")
        checked.append("declared_source_closure_links")
        report.update(passed=True, events=44, admitted_events=len(admitted), held_events=len(held),
                      admitted_event_ids=admitted, held_event_ids=[x["event_id"] for x in held],
                      original_admitted_events=31, original_held_events=13,
                      all44_metadata_and_original_gates_preserved=True,
                      declared_admission_dependencies=len(declared),
                      declared_independent_review_dependencies=len(reviewed))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    for path, sha in inputs.items():
        if digest(Path(path).read_bytes()) != sha:
            report.update(passed=False, error_type="ValueError", error="Input changed during schema check")
    report.update(checks_completed=checked, input_hashes=inputs)
    with (ROOT/OUTPUT).open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks_completed")}))
    print(json.dumps(dict(path=OUTPUT, sha256=digest((ROOT/OUTPUT).read_bytes()))))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
