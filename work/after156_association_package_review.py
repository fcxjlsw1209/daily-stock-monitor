"""Independent metadata-package verifier; no producer or market-data imports."""
import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after156_association_package_review.py'
TEST='work/test_after156_association_package_review.py'
OUT='work/after156_association_package_review.json'
PACKAGE='outputs/monthly_filing_association157_source/'
REVIEW='work/after156_association_source_review.json'
REVIEW_SHA='32b38df82162dbd20cf1d2241bafde3107d7c8283bd855db29caf9530f820469'
DELAYED='work/after156_delayed_partition.json'
META='outputs/monthly_local_linear70/META.json'
ORIGINAL='outputs/monthly_revenue_pilot_source156/ROWS.json'
CALENDAR='outputs/monthly_paired_target143/CALENDAR.json'
PROTOCOL='work/after156_association_protocol.json'
CLOCK='second_XNYS_session_strictly_after_filingDate_v1'
ERAS=(('early53',0,53),('recent35',53,88),('first88',0,88),('later34',88,122),('all122',0,122))
FLOORS=('first88_fraction','early53_fraction','recent35_fraction','first88_issuers',
        'early53_issuers','recent35_issuers','early53_months','recent35_months')
RULES={
    'version':1,
    'scope':'All 5978 original coordinates in original order, 3904 stocks and 2074 ETFs. Every eligible and rejected row is compared; no denominator or neutral-row deletion.',
    'authority':'Only the pinned independent source review supplies eligibility/reasons and evidence identity. This verifier does not change the declared F-only reconstruction or issuer continuity policy.',
    'rows':'Verify all copied fields, exact original cutoff text and integer nanoseconds, current source event/context IDs, selected snapshot, issuer, assumption category, review link, source checks and full partition object. ETF rows retain empty source and partition fields.',
    'partition':'For every materialized partition, independently recompute labels, ownership and complete 19-day bundles from raw filing dates and forms in the hash-bound delayed catalog. This is metadata only.',
    'coverage':'Recompute the five era summaries and exact eight unchanged floors using unique CIKs and the full original stock denominators. Later34 cannot select or alter the floors.',
    'closure':'Verify source-review input closure is contained in manifest input closure, all manifest inputs and output SHA/byte counts before and after validation, and no source reference points outside the closure. Bind actual package producer and project-module dependency closure.',
    'meaning':'Positive means the package faithfully materializes approved conditional metadata research eligibility. Exact public acceptance, historical vintage, real price validity, actual factors and strategy registration remain unasserted.',
}


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def mapping(rows):
    out={}
    for row in rows:
        i,j=row['signal_index'],row['candidate_index']
        require(type(i) is int and type(j) is int and 0<=i<122 and 0<=j<49,'Invalid original coordinate')
        require((i,j) not in out,'Duplicate coordinate')
        out[i,j]=row
    return out


def coverage(rows,etfs):
    eras={}
    for name,lo,hi in ERAS:
        group=[r for r in rows if lo<=r['signal_index']<hi and r['symbol'] not in etfs]
        chosen=[r for r in group if r['research_eligible']]
        issuers={r['issuer_id'] for r in chosen}
        require(None not in issuers,'Missing usable issuer')
        monthly={i:{r['issuer_id'] for r in chosen if r['signal_index']==i} for i in range(lo,hi)}
        eras[name]=dict(stock_slots=len(group),usable_slots=len(chosen),fraction=len(chosen)/len(group),
            distinct_issuers=len(issuers),months_with_at_least8_issuers=sum(len(v)>=8 for v in monthly.values()),
            minimum_issuers_per_month=min(map(len,monthly.values())))
    f,e,r=eras['first88'],eras['early53'],eras['recent35']
    checks=dict(first88_fraction=f['fraction']>=.5,early53_fraction=e['fraction']>=.4,recent35_fraction=r['fraction']>=.4,
        first88_issuers=f['distinct_issuers']>=12,early53_issuers=e['distinct_issuers']>=8,recent35_issuers=r['distinct_issuers']>=8,
        early53_months=e['months_with_at_least8_issuers']>=40,recent35_months=r['months_with_at_least8_issuers']>=26)
    return dict(schema_version=1,original_candidate_rows=5978,original_stock_slots=3904,
        clock_policy_id=CLOCK,exact_acceptance_certified=False,coverage_floor_passed=all(checks.values()),checks=checks,eras=eras)


def partition_check(row,catalog,sessions):
    part=row['partition'];t=row['signal_session_index'];left,right=t-251,t-42
    require(part['first_day']==left and part['last_day']==right,'Partition window changed')
    bundle=defaultdict(list)
    for event_id in row['event_ids']:
        event=catalog[event_id];anchor=bisect_right(sessions,event['filingDate'])+1
        require(t-271<=anchor<=right,'Raw filing anchor outside fixed envelope')
        bundle[anchor].append(event)
    labels=[];owners=[]
    for day in range(left,right+1):
        active=[a for a in bundle if a<=day<=a+20]
        a=max(active) if active else None
        owners.append(a)
        labels.append('N' if a is None else 'B' if day-a<2 or any(e['form'].endswith('/A') for e in bundle[a]) else 'E')
    full=0
    require([b['anchor'] for b in part['bundles']]==sorted(bundle),'Bundle set changed')
    for stored in part['bundles']:
        a=stored['anchor'];events=bundle[a];amended=any(e['form'].endswith('/A') for e in events)
        owned=[left+k for k,(label,owner) in enumerate(zip(labels,owners)) if label=='E' and owner==a]
        complete=not amended and owned==list(range(a+2,a+21));full+=complete
        require(stored['event_ids']==sorted(e['event_id'] for e in events) and stored['forms']==sorted(e['form'] for e in events),'Bundle raw source differs')
        require(stored['amendment'] is amended and stored['full19'] is complete and stored['owned_E_days']==owned,'Bundle ownership differs')
        require(stored['hypothetical_anchor_date']==sessions[a] and stored['left_clipped']==(a+2<left)
            and stored['right_clipped']==(a+20>right),'Bundle boundary differs')
    counts={c:labels.count(c) for c in 'EBN'}
    require(part['labels']==''.join(labels) and part['owner_anchor_indices']==owners and part['counts']==counts
        and part['full19_original_bundles']==full,'Independent partition mismatch')
    passes=full>=2 and counts['E']>=38 and counts['N']>=80
    require(part['hypothetical_partition_pass'] is passes,'Partition pass changed')
    if row['research_eligible']:require(passes,'Eligible row fails original partition minima')


def validate(meta,original,review,delayed,calendar,rows,gate):
    require(review['passed'] is True and review['research_eligibility_verified'] is True
        and review['actual_grid_verified'] is True and review['exact_acceptance_certified'] is False,'Source review admission')
    require(set(review['coverage_floor_checks'])==set(FLOORS) and all(v is True for v in review['coverage_floor_checks'].values()),'Exact source floors required')
    require(len(meta['dates'])==len(meta['symbols'])==122 and len(set(meta['etfs']))==len(meta['etfs'])==17,'META shape')
    require(meta['dates']==sorted(set(meta['dates'])),'META dates')
    for names in meta['symbols']:
        require(len(names)==len(set(names))==49 and len(set(names)&set(meta['etfs']))==17,'META 32 stocks and 17 ETFs')
    require(len(rows)==5978,'Package must contain all5978 rows')
    stocks,etfs,old=mapping(review['rows']),mapping(review['preserved_etf_slots']),mapping(original)
    require(len(stocks)==3904 and len(etfs)==2074 and not stocks.keys()&etfs.keys()
        and stocks.keys()|etfs.keys()==old.keys(),'Original full denominator')
    require(len(old)==5978,'Original source rows')
    positions={day:i for i,day in enumerate(calendar['sessions'])}
    require(len(positions)==len(calendar['sessions']),'Calendar duplicates')
    catalog={p:{e['event_id']:e for e in s['events']} for p,s in delayed['snapshots'].items()}
    partitions=0
    for index,row in enumerate(rows):
        i,j=divmod(index,49);key=(i,j);stock=key in stocks;origin=(stocks if stock else etfs)[key]
        require(type(row['signal_index']) is int and type(row['candidate_index']) is int
            and (row['signal_index'],row['candidate_index'])==key,'Original row order')
        symbol,day=meta['symbols'][i][j],meta['dates'][i]
        require(row['symbol']==origin['symbol']==old[key]['symbol']==symbol,'Original symbol changed')
        require(row['decision_date']==origin['signal_date']==old[key]['decision_date']==day,'Original decision date')
        require(stock==(symbol not in meta['etfs']),'ETF/stock classification changed')
        require(type(row['research_eligible']) is bool and row['research_eligible'] is origin['research_eligible']
            and row['reasons']==origin['reasons'],'Eligibility or reasons changed')
        require(row['decision_cutoff']==old[key]['decision_cutoff'] and type(row['decision_cutoff_ns']) is int
            and row['decision_cutoff_ns']==old[key]['decision_cutoff_ns'],'Original cutoff changed')
        require(row['signal_session_index']==positions[day] and type(row['signal_session_index']) is int,'Calendar signal index')
        require(row['exact_acceptance_certified'] is False and row['source_vintage']=='retrospectively_retrieved_with_explicit_reconstruction_assumptions','Clock/vintage claim changed')
        require(row['source_review_ref']==dict(path=REVIEW,sha256=REVIEW_SHA,row_kind='stock' if stock else 'etf',signal_index=i,candidate_index=j),'Source review pointer changed')
        base_fields={'signal_index','candidate_index','symbol','decision_date','decision_cutoff','decision_cutoff_ns','research_eligible','reasons',
            'issuer_id','partition','event_ids','context_original_event_ids','source_refs','source_review_ref','source_vintage','exact_acceptance_certified','signal_session_index'}
        if not stock:
            require(set(row)==base_fields,'Unexpected ETF field')
            require(row['research_eligible'] is False and row['issuer_id'] is None and row['partition'] is None
                and row['event_ids']==row['context_original_event_ids']==row['source_refs']==[],'ETF must remain neutral')
            continue
        require(set(row)==base_fields|{'selected_snapshot','assumption_category','source_gate_checks'},'Unexpected stock field')
        require(row['decision_cutoff_ns']==origin['decision_cutoff_ns'] and row['signal_session_index']==origin['signal_session_index'],'Reviewed clock changed')
        require(row['event_ids']==origin['event_ids'] and row['context_original_event_ids']==origin['context_original_event_ids'],'Consumed event IDs changed')
        require(row['selected_snapshot']==origin['selected_snapshot'] and row['assumption_category']==origin['assumption_category']
            and row['source_gate_checks']==origin['checks'],'Source or assumption changed')
        ref=origin['partition_ref'];d=delayed['rows'][ref['row_index']]
        require((d['signal_index'],d['candidate_index'],d['symbol'])==(i,j,symbol),'Delayed pointer mismatch')
        require(row['partition']==d['partition'],'Entire partition differs from reviewed source')
        expected_issuer=delayed['snapshots'][row['selected_snapshot']]['cik'] if row['selected_snapshot'] else None
        require(row['issuer_id']==expected_issuer,'Issuer differs from selected snapshot')
        if row['research_eligible']:require(row['issuer_id'] in origin['expected_ciks'],'Eligible issuer mismatch')
        refs={REVIEW:REVIEW_SHA,ref['path']:ref['sha256']}
        for event in origin['source_evidence_refs']:
            for occurrence in event['occurrences']:
                for path,hash_value in ((occurrence['source_path'],occurrence['source_sha256']),(occurrence['receipt_path'],occurrence['receipt_sha256'])):
                    require(path not in refs or refs[path]==hash_value,'Conflicting expected row reference');refs[path]=hash_value
        require(row['source_refs']==[dict(path=p,sha256=h) for p,h in sorted(refs.items())],'Missing/altered provenance ref')
        if row['partition'] is not None:
            require(hashlib.sha256(row['partition']['labels'].encode()).hexdigest()==ref['labels_sha256'],'Reviewed labels digest')
            partition_check(row,catalog[row['selected_snapshot']],calendar['sessions']);partitions+=1
    expected_gate=coverage(rows,set(meta['etfs']))
    require(gate==expected_gate and gate['coverage_floor_passed'],'Independent five-era/eight-floor mismatch')
    require(gate['checks']==review['coverage_floor_checks'],'Source review floor mismatch')
    for name,era in gate['eras'].items():
        source=review['coverage'][name]
        for field,key in [('stock_slots','original_stock_slots'),('usable_slots','research_eligible_slots'),
            ('fraction','fraction'),('distinct_issuers','distinct_issuers'),('months_with_at_least8_issuers','months_with_at_least8_issuers')]:
            require(era[field]==source[key],'Source review coverage mismatch')
    return dict(rows_verified=len(rows),stock_rows_verified=len(stocks),etf_rows_verified=len(etfs),
        partitions_verified=partitions,coverage=expected_gate)


class Inputs:
    def __init__(self):self.hashes={}
    def name(self,value):
        p=Path(value)
        if p.is_absolute():p=p.relative_to(ROOT)
        require('..' not in p.parts and not (ROOT/p).is_symlink() and (ROOT/p).resolve().is_relative_to(ROOT),'Unsafe input path')
        return str(p)
    def bind(self,path,expected=None):
        p=self.name(path);h=sha(ROOT/p)
        require(expected is None or expected==h,'Changed input: '+p)
        require(p not in self.hashes or self.hashes[p]==h,'Input changed during check: '+p)
        self.hashes[p]=h;return p
    def obj(self,path,expected=None):return json.loads((ROOT/self.bind(path,expected)).read_text())
    def closure(self,hashes):
        for path,value in hashes.items():self.bind(path,value)
    def recheck(self):
        for p,h in list(self.hashes.items()):self.bind(p,h)


def audit(manifest_sha):
    inputs=Inputs();review=inputs.obj(REVIEW,REVIEW_SHA);inputs.closure(review['input_hashes'])
    manifest=inputs.obj(PACKAGE+'MANIFEST.json',manifest_sha)
    require(manifest.get('schema_version')==1 and manifest.get('source_only') is True and manifest.get('strategy_registration') is False
        and manifest.get('clock_policy_id')==CLOCK and manifest.get('exact_acceptance_certified') is False,'Manifest claims')
    require(manifest['source_review']==dict(path=REVIEW,sha256=REVIEW_SHA),'Manifest review binding')
    require(all(type(v) is int and v==0 for v in manifest['activity'].values()),'Package must remain metadata-only')
    inputs.closure(manifest['input_hashes'])
    declared={inputs.name(p):h for p,h in manifest['input_hashes'].items()}
    require(all(declared.get(inputs.name(p))==h for p,h in review['input_hashes'].items()),'Missing source-review closure')
    require(declared.get(REVIEW)==REVIEW_SHA,'Missing source review hash')
    for path in (META,ORIGINAL,DELAYED,CALENDAR,PROTOCOL,
                 'work/after156_association_package.py','work/monthly_filing_association157_runner.py',
                 'work/monthly_filing_association157_lifecycle.py','work/complete_monthly_same_report145.py',
                 'work/monthly_forward_guidance132_runner.py','work/monthly_stock_comp149_runner.py',
                 'work/monthly_revenue_persistence156_runner.py','work/after156_association_measure.py'):
        require(path in declared,'Missing mandatory source/producer module: '+path)
    require(set(manifest['outputs'])=={'ROWS.json','COVERAGE_GATE.json'},'Output set differs')
    for name,record in manifest['outputs'].items():
        inputs.bind(PACKAGE+name,record['sha256']);require((ROOT/PACKAGE/name).stat().st_size==record['bytes'],'Output byte count')
    meta,original,delayed,calendar=[inputs.obj(p) for p in (META,ORIGINAL,DELAYED,CALENDAR)]
    rows,gate=[inputs.obj(PACKAGE+p) for p in ('ROWS.json','COVERAGE_GATE.json')]
    validation=validate(meta,original,review,delayed,calendar,rows,gate)
    for row in rows:
        for ref in row['source_refs']:require(declared.get(inputs.name(ref['path']))==ref['sha256'],'Row ref outside manifest closure')
    for p in (SELF,TEST,'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):inputs.bind(p)
    inputs.recheck()
    return inputs,validation


def main():
    p=argparse.ArgumentParser();p.add_argument('--rules-only',action='store_true');p.add_argument('--write',action='store_true')
    p.add_argument('--rules-sha256');p.add_argument('--script-sha256');p.add_argument('--manifest-sha256');args=p.parse_args()
    tests=unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module('test_after156_association_package_review'))
    result=unittest.TextTestRunner(verbosity=1).run(tests);require(result.wasSuccessful(),'Synthetic package validation failed')
    script_sha=sha(ROOT/SELF);rules_sha=digest(RULES)
    if args.rules_only:
        require(not args.write,'Rules-only cannot write')
        print(json.dumps(dict(rules_sha256=rules_sha,script_sha256=script_sha,test_sha256=sha(ROOT/TEST),synthetic_tests=result.testsRun)));return
    require(args.rules_sha256==rules_sha and args.script_sha256==script_sha and args.manifest_sha256,'Bind rules, checker and package manifest before metadata')
    inputs,validation=audit(args.manifest_sha256)
    report=dict(schema_version=1,stage='INDEPENDENT_FILING_ASSOCIATION_PACKAGE_REVIEW',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,research_eligibility_verified=True,actual_grid_verified=True,clock_policy_id=CLOCK,exact_acceptance_certified=False,
        candidate_cells=5978,stock_cells=3904,etf_cells=2074,
        rules=RULES,rules_sha256=rules_sha,script_sha256=script_sha,input_hashes=dict(sorted(inputs.hashes.items())),
        checks=dict(package_matches_source_review=True,original_5978_grid_preserved=True,all_output_and_input_hashes_verified=True,
            stock_3904_etf_2074_preserved=True,all_rows_preserve_evidence_and_eligibility=True,original_cutoffs_preserved=True,
            all_partitions_independently_verified=True,eight_unchanged_coverage_floors_verified=True,no_price_values_read=True),
        validation=dict(validation,synthetic_tests=result.testsRun),
        limitations=[RULES['meaning'],'This validates metadata axes, full partitions, provenance and eligibility, not any actual price/factor/common-finite-mask or strategy result.'],
        activity=dict(price_values_read=0,actual_factors=0,actual_nuisance_estimates=0,orders=0,returns=0,registrations=0,network_requests=0,frozen_files_modified=0))
    if args.write:
        with (ROOT/OUT).open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,validation=report['validation'],output=OUT if args.write else None,output_sha256=sha(ROOT/OUT) if args.write else None),indent=2))


if __name__=='__main__':main()
