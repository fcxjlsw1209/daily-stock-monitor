"""Describe completed round161 outputs; no strategy selection or replay."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_rank161'
PLAN_SHA = '50a25f2f8d5abc452e27b6b5a9ec41fba8ec80ce2917847d70619cfa4feea466'
seen = {}

def read(name):
    p = OUT / name
    raw = p.read_bytes()
    seen[str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def main():
    read('PLAN.json')
    assert seen[str(OUT / 'PLAN.json')] == PLAN_SHA
    completion = read('COMPLETION.json')
    review = read('INDEPENDENT_OUTCOME.json')
    assert completion['plan_sha256'] == review['plan_sha256'] == PLAN_SHA
    assert review['passed'] is True and review['paths_checked'] == 9
    results = read('RESULTS.json')
    assert seen[str(OUT / 'RESULTS.json')] == completion['results_sha256']
    orders = read('PRETRAIN_ORDERS.json')
    names = ('incumbent_local67', 'rank_residence_control', 'rank_trend_challenge')
    nav = {name: {r['date']: r['equity'] for r in read(name + '_same_close_DAILY.json')}
           for name in names}
    for path, digest in seen.items():
        if path in review['artifact_hashes']:
            assert digest == review['artifact_hashes'][path]
    changes = {}
    for name in names[1:]:
        rows = []
        for i, (base, alt) in enumerate(zip(orders[names[0]], orders[name])):
            assert base['date'] == alt['date'] and base['end'] == alt['end']
            if set(base['symbols']) == set(alt['symbols']):
                continue
            start, end = base['date'], base['end']
            bret = nav[names[0]][end] / nav[names[0]][start] - 1
            aret = nav[name][end] / nav[name][start] - 1
            rows.append(dict(index=i, signal_date=start, period_end=end,
                             era='early53' if i < 53 else 'recent35' if i < 88 else 'later34',
                             baseline=base['symbols'], candidate=alt['symbols'],
                             baseline_return=bret, candidate_return=aret,
                             difference_pp=100 * (aret-bret)))
        changes[name] = rows
    report = dict(
        kind='posthoc_description_only_no_reselection', plan_sha256=PLAN_SHA,
        source_hashes=seen, selected=results['selection'],
        metrics={n: {mode: {k: arm['metrics'][k] for k in ('cagr','sharpe','max_drawdown')}
                     for mode, arm in results['arms'][n].items()} for n in names},
        changed_holding_months=changes,
        conclusions=[
            'Both challengers fail the preregistered first88 guards; incumbent retained.',
            'Both challengers also reduce full-period CAGR and Sharpe; no posthoc sign/window/dose adjustment.',
            'This is reused-sample development research with historical price-vintage and estimated-universe limitations.',
            'Future separate source feasibility: reported current debt versus liquid financial assets; no financial factor admitted yet.'
        ])
    with (OUT / 'DIAGNOSTIC_NOTES.json').open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(changed_month_counts={k:len(v) for k,v in changes.items()},
                          worst_changes={k:min(v, key=lambda r:r['difference_pp']) for k,v in changes.items()})))

if __name__ == '__main__':
    main()
