"""Collect successful pure-synthetic logs before the actual150 registration."""
import hashlib
import json
from pathlib import Path
import re
import monthly_financial_coverage150_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage150"
CASES = (
    ("factor", "test_monthly_financial_coverage150_factor.py",
     "/tmp/monthly_financial_coverage150_factor_tests.log"),
    ("selection", "test_monthly_financial_coverage150_selection.py",
     "/tmp/monthly_financial_coverage150_selection_tests.log"),
    ("runner", "test_monthly_financial_coverage150_runner.py",
     "/tmp/monthly_financial_coverage150_runner_tests_final.log"),
    ("lifecycle", "test_monthly_financial_coverage150_lifecycle.py",
     "/tmp/monthly_financial_coverage150_lifecycle_tests_final.log"),
    ("prepare_review", "test_monthly_financial_coverage150_prepare_review.py",
     "PREPARE_REVIEW_TESTS.log"),
    ("outcome_review", "test_monthly_financial_coverage150_outcome_review.py",
     "OUTCOME_REVIEW_TESTS.log"),
)


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    assert not (OUT / "PLAN.json").exists()
    OUT.mkdir(exist_ok=True)
    suites = []
    for name, source, log in CASES:
        origin = Path(log) if log.startswith("/") else OUT / log
        raw = origin.read_bytes()
        text = raw.decode()
        counts = re.findall(r"Ran (\d+) tests? in ", text)
        assert len(counts) == 1 and text.rstrip().endswith("OK")
        dest = OUT / (name.upper() + "_TESTS.log") if log.startswith("/") else origin
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            with dest.open("xb") as stream:
                stream.write(raw)
        suites.append(dict(name=name, count=int(counts[0]),
                           source=ref(ROOT / "work" / source), log=ref(dest)))
    doc = dict(passed=True, synthetic_only=True,
               test_count=sum(x["count"] for x in suites), suites=suites)
    lifecycle.validate_tests(doc, lifecycle.old.Reader(ROOT))
    with (OUT / "TESTS.json").open("x") as stream:
        json.dump(doc, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(dict(**ref(OUT / "TESTS.json"), test_count=doc["test_count"])))
