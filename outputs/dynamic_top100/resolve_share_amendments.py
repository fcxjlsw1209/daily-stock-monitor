"""Resolve only reviewed same-day cover amendments, without certifying cap units."""
import json
from pathlib import Path
import pandas as pd
from normalize_expanded_evidence import normalize

P = Path(__file__).resolve().parent
KEY = ['cik', 'taxonomy', 'concept', 'filed', 'reported_as_of']


def correct_context_dates(raw, evidence):
    out = raw.copy()
    out['original_end'] = out.end
    out['context_source'] = ''
    for fix in evidence:
        if fix.get('verification') != 'original_cover_date' or not fix.get('source'):
            raise ValueError('Missing original cover date evidence')
        mask = pd.Series(True, index=out.index)
        for key in ['cik', 'taxonomy', 'concept', 'filed', 'accn']:
            mask &= out[key].eq(fix[key])
        mask &= out.end.eq(fix['observed_end'])
        if not mask.any() or not out.loc[mask, 'val'].eq(fix['val']).all():
            raise ValueError('Context correction input changed')
        if fix['corrected_end'] > fix['filed']:
            raise ValueError('Future observation date')
        out.loc[mask, 'end'] = fix['corrected_end']
        out.loc[mask, 'context_source'] = fix['source']
    return out


def resolve(events, raw, evidence):
    out = events.copy()
    out['original_conflict'] = out.conflict
    out['amendment_source'] = ''
    for fix in evidence:
        if fix.get('verification') != 'explicit_cover_correction' or not fix.get('source'):
            raise ValueError('Missing reviewed amendment evidence')
        mask = pd.Series(True, index=out.index)
        rmask = pd.Series(True, index=raw.index)
        for key in KEY:
            mask &= out[key].eq(fix[key])
            rmask &= raw['end' if key == 'reported_as_of' else key].eq(fix[key])
        selected = raw.loc[rmask]
        expected = {fix['original_accession'], fix['amendment_accession']}
        if mask.sum() != 1 or set(selected.accn) != expected:
            raise ValueError('Amendment input identity changed or missing')
        if set(selected.val) != set(fix['expected_values']):
            raise ValueError('Amendment input values changed')
        amended = selected.loc[selected.accn.eq(fix['amendment_accession'])]
        if not amended.form.eq('10-Q/A').all() or not amended.val.eq(fix['corrected_shares']).all():
            raise ValueError('Amendment does not confirm replacement')
        if not out.loc[mask, 'conflict'].all():
            raise ValueError('Expected unresolved source conflict')
        out.loc[mask, 'shares'] = fix['corrected_shares']
        out.loc[mask, 'conflict'] = False
        out.loc[mask, 'amendment_source'] = fix['source']
    return out


def reviewed_events():
    evidence = json.loads((P/'pilot_identity/verified_share_amendments.json').read_text())
    dates = json.loads((P/'pilot_identity/verified_share_context_dates.json').read_text())
    events = pd.read_csv(P/'expanded_sec/disclosure_events_unverified.csv.gz', dtype={'cik': str})
    frames = []
    for cik in sorted({fix['cik'] for fix in evidence + dates}):
        fix = {'cik': cik}
        d = pd.read_csv(P/'expanded_sec'/f"{fix['cik']}_shares.csv")
        d['cik'] = fix['cik']
        d['source_file'] = f"expanded_sec/{fix['cik']}_shares.csv"
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    corrected = correct_context_dates(raw, dates)
    target_ciks = {fix['cik'] for fix in dates}
    rebuilt = normalize(corrected[corrected.cik.isin(target_ciks)])
    events = pd.concat([events[~events.cik.isin(target_ciks)], rebuilt], ignore_index=True)
    result = resolve(events, corrected, evidence)
    return result, corrected


def main():
    result, corrected = reviewed_events()
    corrected.to_csv(P/'expanded_sec/context_corrected_raw_UNVERIFIED.csv.gz', index=False)
    result.to_csv(P/'expanded_sec/disclosure_events_amended_UNVERIFIED.csv.gz', index=False)
    report = dict(reviewed_resolutions=int(result.amendment_source.ne('').sum()),
                  reviewed_context_date_corrections=int(corrected.context_source.ne('').sum()),
                  remaining_conflicts=int(result.conflict.sum()), usable_for_backtest=False,
                  original_input_preserved=True, availability_preserved=True,
                  note='Separate evidence view; not an automatic replacement for the admission pipeline.')
    (P/'SHARE_AMENDMENT_AUDIT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
