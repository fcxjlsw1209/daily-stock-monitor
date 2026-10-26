"""Prepare distinct153 lifecycle sources from pinned152, without running them."""
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "monthly_financial_coverage152_lifecycle.py":
        "7f63e535b9595d8404dd3e92a666bf1c433fe7a69483ab2c962508932311399a",
    "test_monthly_financial_coverage152_lifecycle.py":
        "a558e5c9f213383e00a60ac2c3ef0256c5bf86eeee5be690e5d5ccda77247dfd",
}
MAPPING = {
    "monthly_financial_coverage152": "monthly_financial_coverage153",
    "monthly_financial_coverage151": "monthly_financial_coverage152",
    "prepare_financial_coverage152": "prepare_financial_coverage153",
    "prior151": "prior152", "Prior151": "Prior152",
    "finalized151": "finalized152", "Finalized151": "Finalized152",
    "Final151": "Final152", "Exact151": "Exact152",
    "Registered152": "Registered153", "completed152": "completed153",
    "PRE152_": "PRE153_", "five-arm": "six-arm",
    "round=152": "round=153", "== 152": "== 153", "== 151": "== 152",
    "== 957": "== 962", "== 962": "== 968",
    "Expected151 completed rounds/957 trials": "Expected152 completed rounds/962 trials",
    "Final152 rounds/962 trials": "Final153 rounds/968 trials",
    "6f8aa84dadee4d2499e683ecf93a6e82b09ea4f54ec05df5047ef19b8bafcac4":
        "b507b39c3b7362080e954decf95134e487ce43067d6eab3096b25f22c502152d",
    "7dd764a81ff112d2706e4f4216685832830e899ca19af65b77e5dbc5a6648cf5":
        "7b9925053304e38b94e8ad386e0dc74f7d220a26b9884bb3f99e189365e23016",
    "primary_challenger=r.NAMES[4]": "primary_challenger=r.NAMES[5]",
    "('execution_paths_verified',15)": "('execution_paths_verified',18)",
    "('total_monthly_orders_verified',610)": "('total_monthly_orders_verified',732)",
    "('financial_coverage_feature_matrices_verified',8)": "('financial_coverage_feature_matrices_verified',10)",
    "trials=5,independent_execution_paths=15": "trials=6,independent_execution_paths=18",
    "execution_paths_verified=15": "execution_paths_verified=18",
    "total_monthly_orders_verified=610": "total_monthly_orders_verified=732",
    "financial_coverage_feature_matrices_verified=8": "financial_coverage_feature_matrices_verified=10",
    "range(150)": "range(151)", "trials=57": "trials=56", "trials=58": "trials=57",
    "all_45_paths": "all_54_paths",
}


def main():
    expression = re.compile("|".join(re.escape(k) for k in sorted(MAPPING, key=len, reverse=True)))
    products = {}
    for name, digest in PINS.items():
        raw = (ROOT / "work" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        text = expression.sub(lambda m: MAPPING[m.group()], raw.decode())
        if not name.startswith("test_"):
            # Keep historical150/151 checks and add152; global prior-name
            # relocation above applies to the immediate predecessor only.
            text = text.replace(
                "'prior152_batch2_earnings_cache_and_mask_verified', 'batch2_source_records_and_holds_preserved'",
                "'prior151_batch2_earnings_cache_and_mask_verified', 'batch2_source_records_and_holds_preserved',\n"
                "                 'prior152_batch3_earnings_cache_and_mask_verified', 'batch3_source_records_and_holds_preserved'")
            start = text.index("def verify_coverage_sources(")
            end = text.index("\ndef prospective_review(", start)
            text = text[:start] + """def verify_coverage_sources(reader, supplied, previous, prior):
    batches = ('batch1', 'batch2', 'batch3', 'batch4')
    require(set(supplied) == set(batches), 'Exact four source batches required')
    require(set(prior['coverage_sources']) == set(batches[:-1]), 'Prior152 source scope changed')
    for batch in batches[:-1]:
        same_references(reader, supplied[batch], prior['coverage_sources'][batch],
                        'Frozen prior152 source reference changed')
    objects = {batch:verify_coverage(reader, supplied[batch], previous, batch)
               for batch in batches}
    r.combine_sources(*(objects[batch]['dataset'] for batch in batches), supplied)
    return objects

""" + text[end:]
        else:
            text = text.replace(
                "prior152_batch2_earnings_cache_and_mask_verified=True,\n"
                "                batch2_source_records_and_holds_preserved=True)",
                "prior151_batch2_earnings_cache_and_mask_verified=True,\n"
                "                batch2_source_records_and_holds_preserved=True,\n"
                "                prior152_batch3_earnings_cache_and_mask_verified=True,\n"
                "                batch3_source_records_and_holds_preserved=True)")
        target = name.replace("152", "153")
        compile(text, target, "exec")
        products[target] = text
    for name, text in products.items():
        target = ROOT / "work" / name
        with target.open("x") as stream:
            stream.write(text)
        print(name, hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
