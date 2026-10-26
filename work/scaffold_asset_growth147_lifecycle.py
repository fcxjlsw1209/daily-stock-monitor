"""Create new, unfrozen lifecycle scaffolding; never rewrite prior round evidence."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "work/monthly_cash_quality146_lifecycle.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == "b9abaccf378bfaa90936e548a5fc0e374188bb078599879a4deba4be3e239ee0"
text = source.read_text()
text = text.replace("cash-quality", "asset-growth").replace("cash_quality146", "asset_growth147")
text = text.replace("PRE146_", "PRE147_").replace("round=146", "round=147")
text = text.replace('"Already completed146"', '"Already completed147"')
text = text.replace('"Expected145 completed rounds/935 trials"', '"Expected146 completed rounds/939 trials"')
text = text.replace("len(items) == 145", "len(items) == 146").replace('== 935', '== 939')
text = text.replace('"monthly_same_report145"', '"monthly_cash_quality146"')
text = text.replace('outputs/monthly_same_report145/', 'outputs/monthly_cash_quality146/')
text = text.replace(
    'PROPOSAL_SHA = "585fea23faef22c724b7977273f3d3c5994c673198c0e88115f12288accdf51a"',
    'PROPOSAL_SHA = "PENDING_BEFORE_REGISTRATION"')
text = text.replace(
    'PREVIOUS_PLAN = "23135db73d4721fa4ee6d77a32e029825b2be5ac3792f2061ab4eae648c00bfc"',
    'PREVIOUS_PLAN = "23807fe79e5cb9bd6a21273ec7ade26d8a6cb36aab18715e2e052fe1437e81f2"')
text = text.replace(
    'PREVIOUS_DONE = "3819dd665abf5bfc4b389b04546f76b27fed3463d364d707b761e2dab3e09bed"',
    'PREVIOUS_DONE = "23c806429ece2037612503f309e09e348bf7cf0fd26258fe867096008773676e"')
anchor = '    # Continuation is deliberately mutable and therefore a checked precondition,'
insertion = '''    use = proposal["sources"]["asset_use_policy"]
    use_review = proposal["sources"]["asset_use_review"]
    for ref in (use, use_review):
        require(reader.hash(ref["path"]) == ref["sha256"], "Asset-use evidence changed")
        document = reader.read(ref["path"])
        for field in ("input_hashes", "source_hashes", "bindings"):
            if field in document:
                reader.verify(document[field])
    require(reader.read(use_review["path"]).get("passed") is True,
            "Independent asset-use review required")
'''
assert anchor in text
text = text.replace(anchor, insertion + anchor)
text = text.replace(
    'hashes=hashes, inputs=inputs, source_policy=proposal["sources"]["policy"],',
    'hashes=hashes, inputs=inputs, source_policy=proposal["sources"]["policy"],\n'
    '        asset_use_policy=use, asset_use_review=use_review,')
path = ROOT / "work/monthly_asset_growth147_lifecycle.py"
with path.open("x") as handle:
    handle.write(text)

tests = (ROOT / "work/test_monthly_cash_quality146_lifecycle.py").read_text()
tests = tests.replace("cash_quality146", "asset_growth147").replace("PRE146_", "PRE147_")
tests = tests.replace("range(144)", "range(145)").replace("trials=71", "trials=69")
tests = tests.replace("trials=72", "trials=70").replace('"monthly_same_report145"', '"monthly_cash_quality146"')
with (ROOT / "work/test_monthly_asset_growth147_lifecycle.py").open("x") as handle:
    handle.write(tests)
print("Created unfrozen147 lifecycle and synthetic state tests; no registration or numerical work.")
