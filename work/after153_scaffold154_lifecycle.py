"""Create distinct round154 lifecycle sources without executing research."""
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "monthly_financial_coverage153_lifecycle.py":
        "f6fb17acc90bd131d88aecd68936ceea5dd5d139dcfcb12b9d1c642686ac8721",
    "test_monthly_financial_coverage153_lifecycle.py":
        "863ebd1de17eca736ef8bbaccfc3e662d9a7c7580737a9cf669afacfa80b73b2",
}
MAPPING = {
    "monthly_financial_coverage153": "monthly_financial_coverage154",
    "monthly_financial_coverage152": "monthly_financial_coverage153",
    "prepare_financial_coverage153": "prepare_financial_coverage154",
    "prior152": "prior153", "Prior152": "Prior153",
    "finalized152": "finalized153", "Finalized152": "Finalized153",
    "Final153": "Final154", "Exact152": "Exact153",
    "Registered153": "Registered154", "completed153": "completed154",
    "PRE153_": "PRE154_", "six-arm": "seven-arm",
    "round=153": "round=154", "== 153": "== 154", "== 152": "== 153",
    "== 962": "== 968", "== 968": "== 975",
    "Expected152 completed rounds/962 trials": "Expected153 completed rounds/968 trials",
    "Final153 rounds/968 trials": "Final154 rounds/975 trials",
    "b507b39c3b7362080e954decf95134e487ce43067d6eab3096b25f22c502152d":
        "3ced8088fa5fed13ec29591e581ed0c38ed5f88916acff5b6a3f9e33b50b190b",
    "7b9925053304e38b94e8ad386e0dc74f7d220a26b9884bb3f99e189365e23016":
        "6828b1dfd341f34cccc77d3d20bb3135d0fb28a3a2f80d8eb6e29c52157dc560",
    "primary_challenger=r.NAMES[5]": "primary_challenger=r.NAMES[6]",
    "('execution_paths_verified',18)": "('execution_paths_verified',21)",
    "('total_monthly_orders_verified',732)": "('total_monthly_orders_verified',854)",
    "('financial_coverage_feature_matrices_verified',10)":
        "('financial_coverage_feature_matrices_verified',12)",
    "trials=6,independent_execution_paths=18": "trials=7,independent_execution_paths=21",
    "execution_paths_verified=18": "execution_paths_verified=21",
    "total_monthly_orders_verified=732": "total_monthly_orders_verified=854",
    "financial_coverage_feature_matrices_verified=10":
        "financial_coverage_feature_matrices_verified=12",
    "range(151)": "range(152)", "trials=56": "trials=56", "trials=57": "trials=57",
    "all_54_paths": "all_63_paths",
    "for name in names),54)": "for name in names),63)",
}


def main():
    expression = re.compile("|".join(re.escape(k) for k in sorted(MAPPING, key=len, reverse=True)))
    products = {}
    for name, digest in PINS.items():
        raw = (ROOT / "work" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        text = expression.sub(lambda m: MAPPING[m.group()], raw.decode())
        if not name.startswith("test_"):
            text = text.replace(
                "'prior153_batch3_earnings_cache_and_mask_verified', 'batch3_source_records_and_holds_preserved'",
                "'prior152_batch3_earnings_cache_and_mask_verified', 'batch3_source_records_and_holds_preserved',\n"
                "                 'prior153_batch4_earnings_cache_and_mask_verified', 'batch4_source_records_and_holds_preserved'")
            text = text.replace(
                "batches = ('batch1', 'batch2', 'batch3', 'batch4')",
                "batches = ('batch1', 'batch2', 'batch3', 'batch4', 'batch5')")
            text = text.replace("Exact four source batches required", "Exact five source batches required")
        else:
            text = text.replace(
                "prior153_batch3_earnings_cache_and_mask_verified=True,\n"
                "                batch3_source_records_and_holds_preserved=True)",
                "prior152_batch3_earnings_cache_and_mask_verified=True,\n"
                "                batch3_source_records_and_holds_preserved=True,\n"
                "                prior153_batch4_earnings_cache_and_mask_verified=True,\n"
                "                batch4_source_records_and_holds_preserved=True)")
        target = name.replace("153", "154")
        compile(text, target, "exec")
        products[target] = text
    for name, text in products.items():
        target = ROOT / "work" / name
        with target.open("x") as stream:
            stream.write(text)
        print(name, hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
