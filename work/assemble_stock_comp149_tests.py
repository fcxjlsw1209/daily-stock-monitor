"""Assemble immutable successful synthetic logs before round149 registration."""
import hashlib
import json
from pathlib import Path
import re
import monthly_stock_comp149_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/monthly_stock_comp149"
CASES = (
    ("factor", 25, "test_monthly_stock_comp149_factor.py", "/tmp/monthly_stock_comp149_factor_tests_final.log"),
    ("runner", 30, "test_monthly_stock_comp149_runner.py", "/tmp/monthly_stock_comp149_runner_tests_final_v2.log"),
    ("lifecycle", 11, "test_monthly_stock_comp149_lifecycle.py", "/tmp/monthly_stock_comp149_lifecycle_tests_final_v3.log"),
    ("proposal", 4, "test_prepare_stock_comp149_proposal.py", "/tmp/monthly_stock_comp149_proposal_tests_final_v2.log"),
    ("prepare_review", 61, "test_monthly_stock_comp149_prepare_review.py", "PREPARE_REVIEW_TESTS.log"),
    ("outcome_review", 52, "test_monthly_stock_comp149_outcome_review.py", "OUTCOME_REVIEW_TESTS.log"),
)


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert not (OUT/"PLAN.json").exists()
    suites = []
    for name, count, source, log in CASES:
        origin = Path(log) if log.startswith("/") else OUT/log
        raw = origin.read_bytes()
        text = raw.decode()
        assert len(re.findall(r"Ran " + str(count) + r" tests? in ", text)) == 1
        assert text.rstrip().endswith("OK")
        dest = OUT/(name.upper()+"_TESTS.log") if log.startswith("/") else origin
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            with dest.open("xb") as stream:
                stream.write(raw)
        suites.append(dict(name=name, count=count, source=ref(ROOT/"work"/source), log=ref(dest)))
    doc = dict(passed=True, synthetic_only=True, test_count=sum(x["count"] for x in suites), suites=suites)
    lifecycle.validate_tests(doc, lifecycle.old.Reader(ROOT))
    with (OUT/"TESTS.json").open("x") as stream:
        json.dump(doc, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(dict(**ref(OUT/"TESTS.json"), test_count=doc["test_count"])))


if __name__ == "__main__":
    main()
