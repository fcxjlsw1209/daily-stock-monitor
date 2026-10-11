"""Reconstruct a pre-2010 seed; later lists are comparison evidence only.

The bounded pre-start substitutions remain explicit research assumptions where
an exact index effective date is not recovered. This does not issue certificates.
"""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent

def reconstruct(holdings, replacements, start='2010-01-04'):
    seed=set(holdings)
    if len(seed)!=len(holdings):raise ValueError('Duplicate seed security')
    for row in replacements:
        if row['known_before']>=start or row['completed_before']>=start:
            raise ValueError('Pre-start substitution not established before start')
        if row['deletion'] not in seed or row['addition'] in seed:
            raise ValueError('Invalid baseline replacement')
        seed.remove(row['deletion']);seed.add(row['addition'])
    return sorted(seed)

def main():
    base=json.loads((P/'sp100_2009_holdings_corrob.json').read_text())
    if hashlib.sha256((P/base['source_file']).read_bytes()).hexdigest()!=base['sha256']:
        raise ValueError('Baseline source changed')
    replacements=[
      dict(deletion='COV',addition='MET',known_before='2009-05-29',completed_before='2009-06-06',
           evidence='sources/sp_baseline/met_cov_2009_facts.json',status='primary_exact_effective_date'),
      dict(deletion='WYE',addition='NWSA',known_before='2009-10-03',completed_before='2009-12-31',
           evidence='sources/sp_baseline/nwsa_wye_2009_facts.json',status='pre_start_completion_assumption_exact_index_date_unresolved'),
      dict(deletion='SGP',addition='FCX',known_before='2009-10-30',completed_before='2009-12-31',
           evidence='sources/sp_baseline/fcx_sgp_2009_facts.json',status='pre_start_completion_assumption_exact_index_date_unresolved')]
    seed=reconstruct(base['tickers_inferred_from_printed_company_names'],replacements)
    later=set(json.loads((P/'SP100_INITIAL_CORROBORATION.json').read_text())['symbols'])
    sources=[base['source_file']]+[r['evidence'] for r in replacements]
    result=dict(as_of='2010-01-04',constructed_members=seed,count=len(seed),
      construction='March2009 OEF holdings plus three pre-start announced replacements; no later-list seed input',
      replacements=replacements,sources=[dict(file=f,sha256=hashlib.sha256((P/f).read_bytes()).hexdigest()) for f in sources],
      later_list_comparison=dict(only_constructed=sorted(set(seed)-later),only_later=sorted(later-set(seed))),
      verified=False,limitations=['Representative-sampling report plus replacement evidence, not an official index snapshot.',
      'WYE and SGP pre-start completion inferred from completed mergers and subsequent corroboration; exact index sessions still unresolved.',
      'The later stale list is never passed into reconstruction.'])
    (P/'SP100_INITIAL_RECONSTRUCTION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(count=len(seed),later_list_comparison=result['later_list_comparison'],verified=False)))
if __name__=='__main__':main()
