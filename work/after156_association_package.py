"""Materialize reviewed metadata on the unchanged5978-cell strategy axis.

Requires the explicit independent source-review digest. No price values,
scores, orders, model estimates or return series are accessed. The output
still requires a separate package review before strategy registration.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import sys
import monthly_filing_association157_lifecycle as lifecycle
import monthly_filing_association157_runner as runner

ROOT = Path(__file__).resolve().parents[1]
REVIEW = 'work/after156_association_source_review.json'
DELAYED = 'work/after156_delayed_partition.json'
META = 'outputs/monthly_local_linear70/META.json'
ORIGINAL_ROWS = 'outputs/monthly_revenue_pilot_source156/ROWS.json'
require = lifecycle.require


def construct(meta, original, review, delayed, review_ref):
    """Pure deterministic translation; eligibility may only be copied."""
    require(review.get('passed') is True and review.get('research_eligibility_verified') is True and
            review.get('actual_grid_verified') is True and review.get('exact_acceptance_certified') is False,
            'Positive independent conditional source review required')
    require(all(review.get('coverage_floor_checks', {}).values()) and len(review['coverage_floor_checks']) == 8,
            'All eight predeclared source floors required')
    def index(items):
        result = {(r['signal_index'],r['candidate_index']): r for r in items}
        require(len(result) == len(items), 'Duplicate source coordinate')
        return result
    stocks, etfs, prior = index(review['rows']), index(review['preserved_etf_slots']), index(original)
    require(len(stocks) == 3904 and len(etfs) == 2074 and len(prior) == 5978 and not stocks.keys() & etfs.keys(),
            'Full original stock and ETF denominator required')
    rows=[]
    for i, names in enumerate(meta['symbols']):
        for j, symbol in enumerate(names):
            key=(i,j); old=prior[key]
            require((old['symbol'],old['decision_date']) == (symbol,meta['dates'][i]), 'Original source grid changed')
            origin = stocks[key] if key in stocks else etfs[key]
            require((origin['symbol'],origin['signal_date']) == (symbol,meta['dates'][i]), 'Reviewed source grid changed')
            row = dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=meta['dates'][i],
                decision_cutoff=old['decision_cutoff'],decision_cutoff_ns=old['decision_cutoff_ns'],
                research_eligible=origin['research_eligible'],reasons=deepcopy(origin['reasons']),
                issuer_id=None,partition=None,event_ids=[],context_original_event_ids=[],source_refs=[],
                source_review_ref=dict(review_ref, row_kind='stock' if key in stocks else 'etf',
                                       signal_index=i,candidate_index=j),
                source_vintage='retrospectively_retrieved_with_explicit_reconstruction_assumptions',
                exact_acceptance_certified=False)
            if key in stocks:
                row['signal_session_index']=origin['signal_session_index']
                require(old['decision_cutoff_ns']==origin['decision_cutoff_ns'], 'Reviewed cutoff changed')
                ref=origin['partition_ref']; d=delayed['rows'][ref['row_index']]
                require((d['signal_index'],d['candidate_index'],d['symbol'])==(i,j,symbol), 'Partition reference identity')
                row['partition']=deepcopy(d['partition'])
                if row['partition']:
                    require(hashlib.sha256(row['partition']['labels'].encode()).hexdigest()==ref['labels_sha256'],
                            'Partition labels changed')
                row['event_ids']=deepcopy(origin['event_ids'])
                row['context_original_event_ids']=deepcopy(origin['context_original_event_ids'])
                row['selected_snapshot']=origin['selected_snapshot']
                row['assumption_category']=origin['assumption_category']
                if origin['selected_snapshot']:
                    row['issuer_id']=delayed['snapshots'][origin['selected_snapshot']]['cik']
                    if row['research_eligible']:
                        require(row['issuer_id'] in origin['expected_ciks'], 'Reviewed issuer does not match snapshot')
                refs = {review_ref['path']:review_ref['sha256'], ref['path']:ref['sha256']}
                for event in origin['source_evidence_refs']:
                    for occurrence in event['occurrences']:
                        for p,h in ((occurrence['source_path'],occurrence['source_sha256']),
                                    (occurrence['receipt_path'],occurrence['receipt_sha256'])):
                            require(p not in refs or refs[p]==h, 'Contradictory row source digest')
                            refs[p]=h
                row['source_refs']=[dict(path=p,sha256=h) for p,h in sorted(refs.items())]
                row['source_gate_checks']=deepcopy(origin['checks'])
            else:
                require(symbol in meta['etfs'] and origin['research_eligible'] is False, 'ETF feature must stay neutral')
                # Full calendar indexes are injected below from the pinned calendar.
            rows.append(row)
    require(set(prior)==stocks.keys() | etfs.keys() and len(rows)==5978, 'No lost or added coordinates')
    return rows


def build(review_sha, root=ROOT):
    reader=lifecycle.old.Reader(root)
    require(lifecycle.old.digest(review_sha) and reader.hash(REVIEW)==review_sha, 'Explicit reviewed source SHA required')
    review=reader.read(REVIEW); inputs=reader.verify(review['input_hashes'])
    require(all(type(n) is int and n==0 for n in review['activity'].values()), 'Source stage must remain metadata-only')
    for key in ('original_122x49_grid_preserved','all_3904_stock_rows_and_2074_etfs_preserved',
                'source_choice_independently_verified','all_partitions_independently_reconstructed',
                'original_cutoffs_and_cross_snapshot_exceptions_verified','issuer_policy_independently_reconstructed',
                'raw_source_reported_bounds_checked','no_price_values_decoded'):
        require(review['checks'].get(key) is True, 'Missing independent check '+key)
    meta,original,delayed = (reader.read(p) for p in (META,ORIGINAL_ROWS,DELAYED))
    require(reader.hash(lifecycle.DESIGN)==lifecycle.DESIGN_SHA and
            reader.hash(lifecycle.SOURCE_CALENDAR)==lifecycle.SOURCE_CALENDAR_SHA,'Design/calendar binding')
    rows=construct(meta,original,review,delayed,dict(path=REVIEW,sha256=review_sha))
    calendar=reader.read(lifecycle.SOURCE_CALENDAR);positions={d:i for i,d in enumerate(calendar['sessions'])}
    for row in rows:
        t=positions[row['decision_date']]
        require('signal_session_index' not in row or row['signal_session_index']==t,'Calendar index changed')
        row['signal_session_index']=t
        if row['research_eligible']:runner.validate_partition(row)
        for ref in row['source_refs']:
            require(reader.hash(ref['path'])==ref['sha256'],'Row source changed')
    gate=runner.coverage_gate(meta,rows)
    require(gate['coverage_floor_passed'],'Translated source package must pass eight unchanged floors')
    for name,item in gate['eras'].items():
        original=review['coverage'][name]
        require(item['stock_slots']==original['original_stock_slots'] and
                item['usable_slots']==original['research_eligible_slots'] and
                item['distinct_issuers']==original['distinct_issuers'] and
                item['months_with_at_least8_issuers']==original['months_with_at_least8_issuers'],
                'Package coverage differs from independent review')
    for p in ('work/after156_association_package.py','work/monthly_filing_association157_runner.py',
              'work/monthly_filing_association157_lifecycle.py'):
        reader.hash(p)
    reader.verify(runner.source_files(root))
    # Bind the actual local producer import closure as well as the runner's
    # declared sources. This reads source bytes, never price values.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(Path(root).resolve()):
                if not path.is_relative_to(Path(root).resolve()/'work/venv'):
                    reader.hash(path)
    reader.verify(dict(reader.seen))
    return rows,gate,dict(reader.seen)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-review-sha',required=True)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    rows,gate,pins=build(args.source_review_sha)
    out=ROOT/lifecycle.SOURCE_ROOT
    if args.write:
        require(not out.exists(),'Source package output already exists; preserve original')
        out.mkdir()
        for name,value in (('ROWS.json',rows),('COVERAGE_GATE.json',gate)):lifecycle.write(out/name,value)
        manifest=dict(schema_version=1,source_only=True,strategy_registration=False,
            clock_policy_id=runner.CLOCK,exact_acceptance_certified=False,
            created_utc=datetime.now(timezone.utc).isoformat(),input_hashes=pins,
            source_review=dict(path=REVIEW,sha256=args.source_review_sha),
            outputs={n:dict(sha256=lifecycle.sha(out/n),bytes=(out/n).stat().st_size)
                     for n in ('ROWS.json','COVERAGE_GATE.json')},
            activity=dict(price_values_read=0,features=0,orders=0,returns=0,nuisance_estimates=0,registrations=0))
        lifecycle.write(out/'MANIFEST.json',manifest)
    print(lifecycle.old.encoded(dict(written=args.write,coverage=gate,
        manifest_sha256=lifecycle.sha(out/'MANIFEST.json') if args.write else None)).decode())
