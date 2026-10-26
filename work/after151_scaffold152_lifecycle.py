"""Create the new152 lifecycle source from the frozen151 template, without replay."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "work/monthly_financial_coverage151_lifecycle.py":
        "1a70df5e28d610d513dbbdc85465e223515be564cb56b37bedc872601fc27cb4",
    "work/test_monthly_financial_coverage151_lifecycle.py":
        "21912b97da5e821467c7196079202feaefca914a146489e6c587f8da8a73b6cb",
}


def source(name):
    raw = (ROOT / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINS[name]
    return raw.decode()


def substitute(text, before, after, count=None):
    assert before in text, before
    if count is not None:
        assert text.count(before) == count, before
    return text.replace(before, after)


def main():
    s = source("work/monthly_financial_coverage151_lifecycle.py")
    for before, after in (
        ("monthly_financial_coverage151", "monthly_financial_coverage152"),
        ("four-arm", "five-arm"), ("PRE151_", "PRE152_"),
        ("prior150", "prior151"),
        ("round=151", "round=152"), ("== 151", "== 152"),
        ("Registered151", "Registered152"), ("completed151", "completed152"),
        ("Final151 rounds/957 trials", "Final152 rounds/962 trials"),
        ("len(items) == 150", "len(items) == 151"),
        ("== 953", "== 957"), ("== 957,\n", "== 962,\n"),
        ("Expected150 completed rounds/953 trials", "Expected151 completed rounds/957 trials"),
        ("monthly_financial_coverage150", "monthly_financial_coverage151"),
        ("== 150", "== 151"), ("finalized150", "finalized151"), ("Finalized150", "Finalized151"),
        ("Prior150", "Prior151"), ("prior150", "prior151"),
        ("'1d14bd0e34fcaaa27be9fe290d5b6e9f27aa882723f5c521eacdd294160f55db'",
         "'6f8aa84dadee4d2499e683ecf93a6e82b09ea4f54ec05df5047ef19b8bafcac4'"),
        ("'28546cec6459a8ca73df5b434d06878ca7bb16008fc0e13b1a6b870cee8ffb92'",
         "'7dd764a81ff112d2706e4f4216685832830e899ca19af65b77e5dbc5a6648cf5'"),
        ("primary_challenger=r.NAMES[3]", "primary_challenger=r.NAMES[4]"),
        ("('execution_paths_verified',12)", "('execution_paths_verified',15)"),
        ("('total_monthly_orders_verified',488)", "('total_monthly_orders_verified',610)"),
        ("('financial_coverage_feature_matrices_verified',6)", "('financial_coverage_feature_matrices_verified',8)"),
        ("trials=4,independent_execution_paths=12", "trials=5,independent_execution_paths=15"),
    ):
        if before in s:
            s = s.replace(before, after)
    # The predecessor and the final total have distinct counts.
    s = substitute(s,
        "len(items) == 151 and sum(item['trials'] for item in items) == 962",
        "len(items) == 151 and sum(item['trials'] for item in items) == 957")
    start = s.index("def verify_coverage_sources(")
    end = s.index("\ndef prospective_review(", start)
    s = s[:start] + '''def verify_coverage_sources(reader, supplied, previous, prior):
    require(set(supplied) == {'batch1', 'batch2', 'batch3'}, 'Exact three source batches required')
    require(set(prior['coverage_sources']) == {'batch1', 'batch2'}, 'Prior151 source scope changed')
    for batch in ('batch1', 'batch2'):
        same_references(reader, supplied[batch], prior['coverage_sources'][batch],
                        'Frozen prior151 source reference changed')
    objects = {batch:verify_coverage(reader, supplied[batch], previous, batch)
               for batch in ('batch1', 'batch2', 'batch3')}
    r.combine_sources(objects['batch1']['dataset'], objects['batch2']['dataset'],
                      objects['batch3']['dataset'], supplied)
    return objects

''' + s[end:]
    s = substitute(s,
        "'prior151_batch1_earnings_cache_and_mask_verified', 'batch1_source_records_and_holds_preserved'",
        "'prior150_batch1_earnings_cache_and_mask_verified', 'batch1_source_records_and_holds_preserved',\n"
        "                 'prior151_batch2_earnings_cache_and_mask_verified', 'batch2_source_records_and_holds_preserved'")
    compile(s, "monthly_financial_coverage152_lifecycle.py", "exec")

    t = source("work/test_monthly_financial_coverage151_lifecycle.py")
    for before, after in (
        ("monthly_financial_coverage151", "monthly_financial_coverage152"),
        ("monthly_financial_coverage150", "monthly_financial_coverage151"),
        ("range(149)", "range(150)"), ("trials=59", "trials=57"), ("trials=60", "trials=58"),
        ("round=151", "round=152"), ("PRE151_", "PRE152_"),
        ("execution_paths_verified=12", "execution_paths_verified=15"),
        ("total_monthly_orders_verified=488", "total_monthly_orders_verified=610"),
        ("financial_coverage_feature_matrices_verified=6", "financial_coverage_feature_matrices_verified=8"),
        ("all_36_paths", "all_45_paths"),
    ):
        t = t.replace(before, after)
    t = substitute(t, "batch1_source_records_and_holds_preserved=True)",
        "batch1_source_records_and_holds_preserved=True,\n"
        "                prior151_batch2_earnings_cache_and_mask_verified=True,\n"
        "                batch2_source_records_and_holds_preserved=True)")
    compile(t, "test_monthly_financial_coverage152_lifecycle.py", "exec")
    for name, text in (("monthly_financial_coverage152_lifecycle.py", s),
                       ("test_monthly_financial_coverage152_lifecycle.py", t)):
        out = ROOT / "work" / name
        with out.open("x") as stream:
            stream.write(text)
        print(name, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
