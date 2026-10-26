"""Metadata-only source binding for activity164; no prices/features/returns read."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    paths = {k: ROOT / v for k,v in {
        'meta': 'outputs/monthly_local_linear70/META.json',
        'training_meta': 'outputs/monthly_architecture_restart/META.json',
        'prices': 'outputs/monthly_2016_extension/PRICES.npz',
        'calendar': 'outputs/monthly_paired_target143/CALENDAR.json',
        'proposal': 'work/activity164_proposal.json',
        'source_probe': 'work/activity164_source_probe.py',
        'novelty_review': 'work/activity164_novelty_review.json',
    }.items()}
    pins = {str(p):sha(p) for p in paths.values()}
    source_plan=ROOT/'outputs/monthly_paired_target143/PLAN.json'
    pins[str(source_plan)]=sha(source_plan)
    raw=json.loads(source_plan.read_text())['raw_source_hashes']
    assert len(raw)==64
    for name,digest in raw.items():
        assert sha(Path(name))==digest
        pins[name]=digest
    novelty=json.loads(paths['novelty_review'].read_text())
    assert novelty['actual_features_computed']==novelty['actual_orders_computed']==novelty['actual_returns_computed']==0
    m=json.loads(paths['meta'].read_text())
    tm=json.loads(paths['training_meta'].read_text())['training']
    cal=json.loads(paths['calendar'].read_text())['sessions']
    assert cal == sorted(set(cal))
    ci={d:i for i,d in enumerate(cal)}
    assert len(m['dates'])==122 and len(tm['dates'])==180
    seen={}; overlaps=0
    for day,ss in list(zip(tm['dates'],tm['symbols']))+list(zip(m['dates'],m['symbols'])):
        assert len(ss)==len(set(ss))==49
        assert day in ci
        if day != '2016-08-17':
            i=ci[day]
            assert i+1<len(cal) and cal[i+1][:7] != day[:7]
        if day in seen:
            assert set(seen[day])==set(ss)
            overlaps+=1
        seen[day]=ss
    with np.load(paths['prices'],allow_pickle=False) as z:
        # Only load date and identity arrays; never close/open or forecasts.
        dates=z['dates'].tolist();symbols=z['symbols'].tolist()
        keys=list(z.files)
    assert dates==sorted(set(dates)) and len(symbols)==len(set(symbols))
    assert {'dates','symbols','close','opening'}<=set(keys)
    assert all(d in ci for d in dates)
    required_names=sorted({s for ss in seen.values() for s in ss})
    missing_names=sorted(set(required_names)-set(symbols))
    # Unused early pool names and absent prices are not invented; kernel marks
    # their observations unavailable and both additions remain neutral.
    result=dict(passed=True,conditional_research_sources_ready=True,
        strict_pit_vintage=False,strict_pit_membership=False,strategy_admitted=False,
        proposal_sha256=sha(paths['proposal']),input_hashes=pins,source_hashes=pins,
        created_utc=datetime.now(timezone.utc).isoformat(),
        actual_features_computed=0,actual_orders_computed=0,actual_returns_computed=0,
        summary=dict(event_months=122,training_months=180,combined_dates=len(seen),
            overlapping_dates=overlaps,all_overlap_sets_equal=True,
            noninitial_dates_verified_monthend=True,price_dates=len(dates),
            price_first=dates[0],price_last=dates[-1],price_symbols=len(symbols),
            historical_names_absent_from_price_panel=missing_names,market_proxy_present='SPY' in symbols),
        limitations=[
            'Historical Top32 and theme17 identities are existing research proxies, not certified all-US PIT membership.',
            'Pre-2016 pool origins include inherited_pre2016_historical_proxy.',
            'All212 adjusted prices and211 paired close/volume observations must be positive finite; current eligibility is unchanged; coverage is measured only after registration.',
            'Activity partial correlation and matched variance persistence use the same support; only the activity arm is promotable; causal association is not established.',
            'Prices were not inspected by this probe; numerical source support remains to be audited after registration.',
            'Prior163 outcomes and later34 have been viewed in earlier studies; new study remains development research.'
        ])
    assert all(sha(Path(p))==h for p,h in pins.items())
    out=ROOT/'work/activity164_design_review.json'
    with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'passed':True,'summary':result['summary'],'sha256':sha(out)}))

if __name__=='__main__':main()
