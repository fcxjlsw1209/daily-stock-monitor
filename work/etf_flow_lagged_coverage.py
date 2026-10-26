"""Check a declared five-session reporting-delay source window; no factor math."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from etf_flow_source_quality import ROOT, BASE, sha, coverage, parse_ishares

def main():
    output = BASE / 'LAGGED_COVERAGE_SUPPLEMENT.json'
    if output.exists():
        raise RuntimeError('Write-once audit already exists')
    prior_path = BASE / 'CALENDAR_ACCOUNTING_AUDIT.json'
    assert sha(prior_path) == '2cf542a5755cab8e99100b36fcf04ca60dd60efc4d3feeaea9a37b559399f482'
    prior = json.loads(prior_path.read_text())
    for name, expected in prior['input_sha256'].items():
        assert sha(ROOT / name) == expected, name
    price_path = ROOT / 'outputs/monthly_2016_extension/PRICES.npz'
    with np.load(price_path, allow_pickle=False) as archive:
        dates = archive['dates'].tolist()
    assert dates == sorted(set(dates))
    lag, changes = 5, 63
    i, j = dates.index('2016-08-17'), dates.index('2026-09-24')
    # Extended source coverage. The eventual query at t would only consume
    # through t-5, never these later rows; no feature is produced here.
    sessions = dates[i - changes - lag:j + 1]
    funds = []
    ssga = json.loads((BASE / 'ssga/FEASIBILITY.json').read_text())
    ish = json.loads((BASE / 'ishares/FEASIBILITY.json').read_text())
    for item in ssga['sources']:
        path = ROOT / item['extraction_file']
        assert sha(path) == item['extraction_sha256']
        funds.append({'symbol': item['symbol'],
                      'coverage': coverage(json.loads(path.read_text())['rows'], sessions, sessions[0], sessions[-1])})
    for item in ish['funds']:
        path = BASE / 'ishares' / item['source_file']
        assert sha(path) == item['source_sha256']
        rows, _ = parse_ishares(path.read_bytes())
        funds.append({'symbol': item['ticker'],
                      'coverage': coverage(rows, sessions, sessions[0], sessions[-1])})
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Source feasibility for a hypothetical fixed5-session lag; no strategy registration or factor selection.',
              'reporting_lag_sessions_examined': lag, 'share_changes_window_sessions': changes,
              'minimum_observation_offset_sessions': -(changes + lag),
              'coverage_start': sessions[0], 'coverage_end': sessions[-1], 'session_count': len(sessions),
              'funds': funds, 'prior_audit_sha256': sha(prior_path),
              'code_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'work/etf_flow_source_quality.py'),
              'publication_lag_verified': False, 'strict_point_in_time_admission': False,
              'no_actual_flows_scores_orders_or_returns': True,
              'notes': ['Five sessions is an explicit possible research assumption, not proof of original historical publication.',
                        'A future feature must separately enforce per-query t-5 cutoff and verified action units.',
                        'Earlier63-session audit is preserved; this supplement adds five preceding observations only.']}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'path': str(output.relative_to(ROOT)), 'sha256': sha(output),
                      'start': sessions[0], 'sessions': len(sessions),
                      'complete_funds': [r['symbol'] for r in funds if r['coverage']['complete_nav_and_shares']]}))

if __name__ == '__main__':
    main()
