"""Bind completed source producers; no extraction or factor calculation."""
from pathlib import Path
import hashlib
import json

root = Path.cwd()


def ref(path, expected=None):
    h = hashlib.sha256((root/path).read_bytes()).hexdigest()
    if expected is not None:
        assert h == expected, path
    return dict(path=path, sha256=h)


pdf = "outputs/monthly_cashflow_pdf_repair_v4"
html = "outputs/monthly_cashflow_html_repair_v2"
producers = {}
for kind, directory, plan, review, review_sha in (
    ("pdf", pdf, "6acd731fb3bcbcbaf0cf5c219c6bc5abaccd45fe7be85b95f79ce25ce707df11",
     pdf+"/INDEPENDENT_VISUAL_CHECK.json",
     "a2f992f228d79ae4ece56ce657d07c0b8c44e5a82f1db1cafd2d7f3e4e164ebe"),
    ("html", html, "0c39bd0aee46b28fb6175a0abfb5dedf39ea1804378f318883af37857296a3c3",
     "work/cashflow_html_repair_v2_actual_review.json",
     "7319711e9f62ea028338620211627750de62b2793f1edcd08d736ac6d732f7f4")):
    producers[kind] = dict(directory=directory, plan_sha256=ref(directory+"/PLAN.json", plan)["sha256"],
        manifest_sha256=ref(directory+"/MANIFEST.json")["sha256"],
        independent_reviews=[ref(review, review_sha)])

value = dict(schema_version=1, stage="reviewed_source_union_inputs",
    source_union_authorized=True, admission_authorized=False, producers=producers,
    accounting_review=ref("work/cashflow_accounting_review_v1.json",
        "ec6d74579b43c7cb8f6fda0c9c4855684ee4e16f334add46836c5854ceaebf9b"))
path = "work/cashflow_source_union_v1_bindings.json"
with (root/path).open("x") as f:
    json.dump(value, f, sort_keys=True, indent=2)
    f.write("\n")
print(json.dumps(ref(path)))
