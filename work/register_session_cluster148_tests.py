"""Bind successful synthetic-suite receipts before round148 registration."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    ("factor", "work/test_monthly_session_cluster148_factor.py",
     "work/session_cluster148_factor_tests.log"),
    ("runner", "work/test_monthly_session_cluster148_runner.py",
     "work/session_cluster148_runner_tests.log"),
    ("prepare_review", "work/test_monthly_session_cluster148_prepare_independent.py",
     "outputs/monthly_session_cluster148/PREPARE_REVIEW_TESTS.log"),
    ("outcome_review", "work/test_monthly_session_cluster148_outcome_independent.py",
     "outputs/monthly_session_cluster148/OUTCOME_REVIEW_TESTS.log"),
    ("lifecycle", "work/test_monthly_session_cluster148_lifecycle.py",
     "work/session_cluster148_lifecycle_tests.log"),
)


def reference(relative):
    return dict(path=relative, sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())


def main():
    destination = ROOT / "outputs/monthly_session_cluster148/TESTS.json"
    assert not (destination.parent / "PLAN.json").exists(), "Already registered"
    suites = []
    for name, source, log in SPECS:
        text = (ROOT / log).read_text()
        counts = re.findall(r"Ran (\d+) tests? in ", text)
        assert len(counts) == 1 and text.rstrip().endswith("OK"), log
        suites.append(dict(name=name, count=int(counts[0]),
                           source=reference(source), log=reference(log)))
    result = dict(passed=True, synthetic_only=True, round=148, suites=suites,
                  test_count=sum(s["count"] for s in suites),
                  created_utc=datetime.now(timezone.utc).isoformat())
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(tests=result["test_count"],
                          manifest=reference(str(destination.relative_to(ROOT))))))


if __name__ == "__main__":
    main()
