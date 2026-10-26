"""Bind original prospective synthetic test logs to their current sources."""
from pathlib import Path
import json
import re
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def ref(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    suites=[]
    for name in ("factor","runner","prepare_review","outcome_review","lifecycle"):
        source="work/test_monthly_cash_quality146_"+name+".py"
        log=("outputs/monthly_cash_quality146/"+name.upper()+"_TESTS.log"
             if name in ("prepare_review","outcome_review")
             else "work/cash_quality146_"+name+"_tests.log")
        text=(ROOT/log).read_text()
        matches=re.findall(r"Ran (\d+) tests? in ",text)
        if len(matches)!=1 or not text.rstrip().endswith("OK"):
            raise ValueError("Missing successful original log: "+name)
        suites.append(dict(name=name,count=int(matches[0]),source=ref(source),log=ref(log)))
    doc=dict(passed=True,synthetic_only=True,round=146,suites=suites,
             test_count=sum(s["count"] for s in suites))
    path=ROOT/"outputs/monthly_cash_quality146/TESTS.json"
    with path.open("x") as f:
        json.dump(doc,f,indent=2);f.write("\n")
    print(json.dumps(dict(tests=doc["test_count"],sha256=ref(str(path.relative_to(ROOT)))["sha256"])))


if __name__=="__main__":
    main()
