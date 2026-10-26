"""Collect the six final synthetic suites before round152 registration."""
import hashlib
import json
from pathlib import Path
import re

import monthly_financial_coverage152_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage152"
CASES = (
    ("factor", "/tmp/monthly_financial_coverage152_factor_tests.log"),
    ("selection", "/tmp/monthly_financial_coverage152_selection_tests.log"),
    ("runner", "/tmp/monthly_financial_coverage152_runner_tests.log"),
    ("lifecycle", "/tmp/monthly_financial_coverage152_lifecycle_tests_initial.log"),
    ("prepare_review", "PREPARE_REVIEW_TESTS.log"),
    ("outcome_review", "OUTCOME_REVIEW_TESTS.log"),
)


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    if (OUT / "PLAN.json").exists():
        raise ValueError("Tests must be assembled before registration")
    OUT.mkdir(exist_ok=True)
    suites = []
    for name, log in CASES:
        origin = Path(log) if log.startswith("/") else OUT / log
        raw = origin.read_bytes()
        text = raw.decode()
        counts = re.findall(r"Ran (\d+) tests? in ", text)
        if len(counts) != 1 or not text.rstrip().endswith("OK"):
            raise ValueError("Incomplete synthetic suite: " + name)
        dest = OUT / (name.upper() + "_TESTS.log") if log.startswith("/") else origin
        if dest.exists():
            if dest.read_bytes() != raw:
                raise ValueError("Existing test log differs: " + name)
        else:
            with dest.open("xb") as stream:
                stream.write(raw)
        suites.append(dict(
            name=name, count=int(counts[0]),
            source=ref(ROOT / "work" / ("test_monthly_financial_coverage152_" + name + ".py")),
            log=ref(dest),
        ))
    result = dict(passed=True, synthetic_only=True,
                  test_count=sum(x["count"] for x in suites), suites=suites)
    lifecycle.validate_tests(result, lifecycle.old.Reader(ROOT))
    with (OUT / "TESTS.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(**ref(OUT / "TESTS.json"), test_count=result["test_count"])))


if __name__ == "__main__":
    main()
