"""Source-only structural corroboration of already extracted recast paragraphs.

Scope and raw hashes were frozen in SPEC before the original extraction.
No amounts are extracted or calculated; table openings serve only as locators.
"""
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SPEC_SHA = "27b5bf62baef91502ecc64ccee8726e39575c15b527d589d55ef727efe27e268"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(e):
    return " ".join(e.get_text(" ", strip=True).split())


assert sha(OUT / "SPEC.json") == SPEC_SHA
spec = json.loads((OUT / "SPEC.json").read_text())
records = []
section = re.compile(r"^(?:INCOME STATEMENTS|COMPREHENSIVE INCOME STATEMENTS|CASH FLOWS STATEMENTS|SEGMENT (?:REVENUE AND OPERATING INCOME|RESULTS)|Segment Revenue Constant Currency Reconciliation)$")
for source in spec["sources"]:
    p = Path(source["path"])
    p = p if p.is_absolute() else ROOT / p
    assert p.is_relative_to(ROOT) and p.resolve() == p
    assert sha(p) == source["sha256"]
    soup = BeautifulSoup(p.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    nodes = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    tables = soup.find_all("table")
    for i, node in enumerate(nodes):
        text = norm(node)
        if "We have recast certain prior period amounts" not in text:
            continue
        titles = [{"text": norm(n), "line": n.sourceline} for n in nodes[:i] if section.match(norm(n))]
        later = [{"text": norm(n), "line": n.sourceline} for n in nodes[i+1:] if section.match(norm(n))]
        parent = node.find_parent("table")
        previous = node.find_previous("table")
        contexts = []
        for role, table in (("ancestor", parent), ("previous", previous)):
            if table is None:
                continue
            openings = [norm(x) for x in table.find_all("p")[:2]]
            contexts.append({"role": role, "table_ordinal_1based": next(j+1 for j,x in enumerate(tables) if x is table),
                             "table_line": table.sourceline, "first_two_paragraphs": openings})
        records.append({"fiscal_year": source["fiscal_year"], "source_path": str(p.relative_to(ROOT)),
                        "source_sha256": source["sha256"], "paragraph_ordinal_1based": i+1,
                        "line": node.sourceline, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "nearest_preceding_section": titles[-1] if titles else None,
                        "nearest_following_section": later[0] if later else None,
                        "table_contexts": contexts})
assert [r["fiscal_year"] for r in records] == [2023,2023,2025,2025,2026]
result = {"spec_sha256": SPEC_SHA, "code_sha256": sha(Path(__file__)), "records": records,
          "scope": "All occurrences of the recast note inside the twelve pre-frozen pages, with structural locators. Not a new revenue extraction.",
          "new_financial_values_extracted": 0, "growth_calculations": 0}
target = OUT / "CONTEXT_EVIDENCE.json"
with target.open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("CONTEXT_SHA", sha(target))
