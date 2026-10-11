"""Versioned source corrections: preserve originals, never guess a scale factor."""
import json
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parent
KEY = ['cik', 'taxonomy', 'concept', 'reported_as_of', 'filed']


def reconcile(frame, corrections):
    d = frame.copy()
    d['original_shares'] = d.shares
    d['original_reported_as_of'] = d.reported_as_of
    d['correction_source'] = ''
    for fix in corrections:
        if fix.get('verification') not in {'original_filing_cover', 'original_filing_statement'} or not fix.get('source'):
            raise ValueError('Correction lacks original evidence')
        mask = pd.Series(True, index=d.index)
        for key in KEY:
            mask &= d[key].eq(fix[key])
        mask &= d.accessions.map(lambda x: fix['accession'] in str(x).split(';'))
        if not mask.any():
            continue
        if not d.loc[mask, 'shares'].eq(fix['observed_shares']).all():
            raise ValueError('Source changed; correction no longer matches')
        if not 0 < fix['corrected_shares'] < float('inf'):
            raise ValueError('Invalid replacement')
        if 'corrected_reported_as_of' in fix:
            corrected_date = fix['corrected_reported_as_of']
            if str(pd.Timestamp(corrected_date).date()) != corrected_date or corrected_date > fix['filed']:
                raise ValueError('Invalid source observation date')
            if d.loc[mask].get('conflict', pd.Series(False)).any():
                raise ValueError('Resolve source conflict before correcting context')
            d.loc[mask, 'reported_as_of'] = corrected_date
        d.loc[mask, 'shares'] = fix['corrected_shares']
        d.loc[mask, 'correction_source'] = fix['source']
    # This threshold requests review, not proof of bad data or automatic rescaling.
    d['scale_review_required'] = d.shares.gt(1e12)
    return d


if __name__ == '__main__':
    from resolve_share_amendments import reviewed_events
    d, _ = reviewed_events()
    corrections = json.loads((P / 'pilot_identity/share_corrections.json').read_text())
    result = reconcile(d, corrections)
    result.to_csv(P / 'expanded_sec/disclosure_events_reconciled_UNVERIFIED.csv.gz', index=False, compression='gzip')
    print('corrections', int(result.correction_source.ne('').sum()), 'remaining scale reviews', int(result.scale_review_required.sum()))
