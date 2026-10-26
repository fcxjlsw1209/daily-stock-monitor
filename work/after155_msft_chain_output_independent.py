"""Independent source-only replay of the MSFT revenue-chain preparation.

No imports from either chain producer; no prices, growth rates, models or returns.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
from decimal import Decimal
import hashlib
from io import StringIO
import json
from pathlib import Path
import unittest

import pandas as pd
from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar


ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/monthly_msft_revenue_chain156/'
OUTPUT = 'work/after155_msft_chain_output_independent.json'
PROPOSAL = 'outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json'
BASIS = 'work/after155_msft_revenue_basis_review.json'
POLICY = 'msft_original_annual_revenue_four_level_chain_v1'
PINS = {
    BASE+'SPEC.json': 'ffa2849c2f3a4a2ded6bc4e3004a64843b45e80cda286d3ca0f74a3e773ea098',
    BASE+'MANIFEST.json': '4d84d4c9957551f931158789218ea44d817bbd961323156f1a28aa0249e92008',
    'work/after155_msft_chain_driver_prereview.json': '19836be44351588ed6373ebfb869866473fe7678ceb3d60945ba8bd7c92a7238',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(name):
    p = Path(name)
    if not p.is_absolute():
        p = ROOT/p
    require(p.is_relative_to(ROOT) and p.resolve() == p, 'Nonlocal path or symlink')
    return p


def digest(name):
    return hashlib.sha256(path(name).read_bytes()).hexdigest()


def load(name):
    return json.loads(path(name).read_text())


def ns(value):
    stamp = pd.Timestamp(value)
    require(stamp.tzinfo is not None, 'Timezone required')
    return stamp.value


def rid(year):
    return 'MSFT_FY%d_original_Q4' % year


def eid(year):
    return 'MSFT_FY%d_to_FY%d' % (year-1, year)


def reference_choice(reports, edges, cutoff, decision_day):
    """Reference decision from source records, independent of producer objects."""
    visible = [r for r in reports if r['available'] <= cutoff]
    if not visible:
        return dict(state='missing', latest=None, years=[], traversed=[], reason='no_known_report')
    times = [r['available'] for r in visible]
    require(len(times) == len(set(times)), 'Ambiguous publication ordering')
    latest = sorted(visible, key=lambda r: r['available'])[-1]
    endpoint = latest['year']
    result = dict(state='valid', latest=endpoint, years=[], traversed=[], reason=None)
    if not latest['valid']:
        result.update(state='invalid', reason='invalid_latest')
        return result
    visible_by_year = {r['year']: r for r in visible}
    require(len(visible_by_year) == len(visible), 'Duplicate annual original')
    required = list(range(endpoint-3, endpoint+1))
    for y in required:
        if y not in visible_by_year:
            result.update(state='missing', reason='missing_own_year_%d' % y)
            return result
    result['years'] = required
    if (date.fromisoformat(decision_day)-date.fromisoformat(latest['end'])).days > 550:
        result.update(state='stale', reason='latest_older_than_550_days')
        return result
    for y in required:
        r = visible_by_year[y]
        require(r['valid'], 'Invalid earlier source')
        require(r['start'] == '%d-07-01' % (y-1) and r['end'] == '%d-06-30' % y,
                'Annual period differs')
        require(r['amount'].is_finite() and r['amount'] > 0, 'Nonpositive/nonfinite source total')
    for y in required[1:]:
        before, after = visible_by_year[y-1], visible_by_year[y]
        edge = edges.get(y)
        if edge is None or edge['available'] > cutoff or any(t > cutoff for t in edge['notes']):
            result.update(state='unresolved', reason='edge_not_known')
            return result
        result['traversed'].append(y)
        require(edge['before'] == rid(y-1) and edge['after'] == rid(y), 'Wrong source edge identity')
        if edge['status'] != 'pass_with_declared_scope':
            result.update(state='blocked', reason=eid(y))
            return result
        if (after['witness'] != before['amount'] or
                after['witness_start'] != before['start'] or after['witness_end'] != before['end'] or
                after['unit'] != before['unit']):
            result.update(state='blocked', reason=eid(y))
            return result
    return result


def fixture():
    reports, edges = [], {}
    for y in range(2015, 2020):
        available = ns('%d-07-25T20:00:00Z' % y)
        reports.append(dict(year=y, available=available, valid=True, start='%d-07-01' % (y-1),
                            end='%d-06-30' % y, amount=Decimal('10'), witness=Decimal('10'),
                            witness_start='%d-07-01' % (y-2), witness_end='%d-06-30' % (y-1),
                            unit=('USD', '1000000')))
        edges[y] = dict(available=available, notes=[available], before=rid(y-1), after=rid(y),
                        status='pass_with_declared_scope')
    return reports, edges


class IndependentChecks(unittest.TestCase):
    def test_exact_cutoff(self):
        r, e = fixture()
        self.assertEqual(reference_choice(r,e,ns('2018-07-25T20:00:00Z'),'2018-07-25')['years'],[2015,2016,2017,2018])

    def test_future_latest_excluded(self):
        r,e = fixture()
        x=reference_choice(r,e,ns('2018-07-25T19:59:59Z'),'2018-07-25')
        self.assertEqual((x['latest'],x['state']),(2017,'missing'))

    def test_no_invalid_latest_fallback(self):
        r,e = fixture(); r[-1]['valid']=False
        x=reference_choice(r,e,ns('2019-07-31T20:00:00Z'),'2019-07-31')
        self.assertEqual((x['latest'],x['state']),(2019,'invalid'))

    def test_blocked_edge(self):
        r,e = fixture(); e[2017]['status']='blocked_known_basis_break'
        x=reference_choice(r,e,ns('2018-07-31T20:00:00Z'),'2018-07-31')
        self.assertEqual((x['state'],x['traversed']),('blocked',[2016,2017]))

    def test_future_note_not_usable(self):
        r,e = fixture(); e[2017]['notes']=[ns('2020-01-01T20:00:00Z')]
        self.assertEqual(reference_choice(r,e,ns('2018-07-31T20:00:00Z'),'2018-07-31')['state'],'unresolved')

    def test_comparative_does_not_replace_original(self):
        r,e=fixture(); r[3]['witness']=Decimal('11')
        self.assertEqual(reference_choice(r,e,ns('2018-07-31T20:00:00Z'),'2018-07-31')['state'],'blocked')
        self.assertEqual(r[2]['amount'],Decimal('10'))

    def test_550_day_ceiling(self):
        r,e=fixture(); r=r[:4]
        end=pd.Timestamp('2018-06-30')
        for days,state in ((550,'valid'),(551,'stale')):
            day=str((end+pd.Timedelta(days=days)).date())
            self.assertEqual(reference_choice(r,e,ns(day+'T20:00:00Z'),day)['state'],state)

    def test_missing_old_original_not_witness_fill(self):
        r,e=fixture(); r=r[1:]
        x=reference_choice(r,e,ns('2018-07-31T20:00:00Z'),'2018-07-31')
        self.assertEqual((x['state'],x['reason']),('missing','missing_own_year_2015'))

    def test_source_order_not_selection_order(self):
        r,e=fixture()
        self.assertEqual(reference_choice(r,e,ns('2019-07-31T20:00:00Z'),'2019-07-31'),
                         reference_choice(list(reversed(r)),e,ns('2019-07-31T20:00:00Z'),'2019-07-31'))


def calendar(frozen):
    # Direct pinned exchange-calendar dependency and explicit closure removal;
    # neither the producer driver nor its calendar helper is imported.
    base=XNYSExchangeCalendar(start=frozen['start'],end=frozen['end']).schedule.copy()
    dates=[str(d.date()) for d in base.index]
    closed={r['date'] for r in load('outputs/index_union/CALENDAR_EXCEPTIONS.json')['closures'] if 'XNYS' in r['markets']}
    closes={day:stamp for day,stamp in zip(dates,base['close']) if day not in closed}
    require(list(closes)==frozen['sessions'],'Independent session axis differs')
    return closes


def build_review(tests_run):
    pins=dict(PINS)
    def bind(name,h):
        name=str(path(name).relative_to(ROOT))
        require(name not in pins or pins[name]==h,'Conflicting hash binding')
        require(digest(name)==h,'Input hash mismatch: '+name)
        pins[name]=h
    for p,h in list(pins.items()): bind(p,h)
    spec=load(BASE+'SPEC.json'); manifest=load(BASE+'MANIFEST.json')
    require(spec['policy_id']==POLICY and manifest['spec_sha256']==PINS[BASE+'SPEC.json'],'Wrong source contract')
    for p,h in spec['hashes'].items(): bind(p,h)
    for p,h in manifest['outputs'].items(): bind(BASE+p,h)
    require(load('work/after155_msft_basis_independent.json')['passed'] is True,'Basis review failed')
    require(load('work/after155_revenue_chain_implementation_review_v2.json')['passed'] is True,'Core review failed')
    closes=calendar(load('outputs/monthly_paired_target143/CALENDAR.json'))
    source=load(PROPOSAL)['records']; basis=load(BASIS)['boundaries']
    actual_reports=load(BASE+'REPORTS.json'); actual_edges=load(BASE+'EDGES.json')
    require(len(source)==len(actual_reports)==12,'Report count')
    require(sorted(r['fiscal_year'] for r in source)==list(range(2015,2027)),'Source years')
    require(len(basis)==len(actual_edges)==11,'Edge count')
    require(len({r['report_id'] for r in actual_reports})==12,'Duplicate report identity')
    require(len({e['edge_id'] for e in actual_edges})==11,'Duplicate edge identity')
    by_id={r['report_id']:r for r in actual_reports}
    normalized=[]; source_by_year={}; report_checks=[]
    for original in source:
        y=original['fiscal_year']; source_by_year[y]=original
        current=original['revenue']['current']; witness=original['revenue']['comparative']
        a=original['availability']; s=original['source']
        require(original['symbol']==s['symbol']=='MSFT' and s['fiscal_year']==y,'Original identity')
        release=a['release_date']; next_day=min(d for d in closes if d>release)
        known=closes[next_day]
        require(a['first_conservative_session_after_release']==next_day and a['precision']=='date_only','Source date')
        require(current['first_conservative_session']==witness['first_conservative_session']==next_day,'Cell date')
        require(a['timezone']=='America/Los_Angeles' and original['currency']['iso_currency']=='USD','Currency/timezone')
        receipt=load(s['receipt']['path'])
        require(receipt['http_status']==200 and receipt['url']==s['expected_url'] and receipt['sha256']==s['raw']['sha256'],'Receipt/source mismatch')
        expected=dict(report_id=rid(y),fiscal_year=y,issuer='MSFT',source_checked=True,source=s,currency='USD',
            measure='consolidated_gaap_total_revenue',known_ns=known.value,known_by=known.isoformat(),release_date=release,
            period_start=current['period_start_derived'],period_end=current['period_end'],current=current['amount_decimal'],
            scale=str(current['scale']),comparative=witness['amount_decimal'],comparative_scale=str(witness['scale']),
            comparative_currency='USD',comparative_measure='consolidated_gaap_total_revenue',
            comparative_period_start=witness['period_start_derived'],comparative_period_end=witness['period_end'],
            comparative_source_report_id=rid(y),proposal_record_ref=dict(path=PROPOSAL,fiscal_year=y),
            current_locator=current['locator'],comparative_locator=witness['locator'],
            current_cell_html_sha256=current['cell_html_sha256'],comparative_cell_html_sha256=witness['cell_html_sha256'],
            row_html_sha256=hashlib.sha256(original['revenue']['revenue_row_html'].encode()).hexdigest(),
            derived_income_table=original['revenue']['derived_table_evidence'],
            archival_vintage='currently_retrieved_original_dated_issuer_release',historical_first_public_bytes_verified=False)
        require(by_id[rid(y)]==expected,'Derived report differs from independent original mapping: '+str(y))
        normalized.append(dict(year=y,available=known.value,valid=True,start=current['period_start_derived'],end=current['period_end'],
            amount=Decimal(current['amount_decimal'])*Decimal(str(current['scale'])),
            witness=Decimal(witness['amount_decimal'])*Decimal(str(witness['scale'])),
            witness_start=witness['period_start_derived'],witness_end=witness['period_end'],unit=('USD',str(current['scale']))))
        require(current['duration_literal']==witness['duration_literal']=='Twelve Months Ended','Duration')
        require(str(current['scale'])==str(witness['scale']),'Witness scale')
        report_checks.append(dict(fiscal_year=y,original_report_id=rid(y),release_date=release,independent_known_by=known.isoformat()))
    edge_lookup={e['edge_id']:e for e in actual_edges}; independent_edges={}
    require(sorted(b['later_fiscal_year'] for b in basis)==list(range(2016,2027)),'Edge coverage')
    for b in basis:
        y=b['later_fiscal_year']; known=by_id[rid(y)]['known_ns']
        require(b['earlier_fiscal_year']==y-1 and ns(b['known_by'])==known,'Boundary identity/time')
        require(b['available_from']==b['known_by']==b['known_no_earlier_than'],'Note clock aliases')
        expected=dict(edge_id=eid(y),earlier_fiscal_year=y-1,later_fiscal_year=y,
            earlier_report_id=rid(y-1),later_report_id=rid(y),known_ns=known,known_by=b['known_by'],
            evidence_known_ns=[ns(e['source_available_at']) for e in b['evidence']],
            status={'pass_with_declared_scope':'checked_reported_total_edge','blocked_known_basis_break':'blocked_known_basis_break'}[b['status']],
            evidence_refs=b['evidence_refs'],evidence_sources=[dict(path=e['source_path'],sha256=e['source_sha256']) for e in b['evidence']],
            context_evidence_ref=b.get('context_evidence_ref'),reason=b['reason'],economic_homogeneity_certified=False)
        require(edge_lookup[eid(y)]==expected,'Derived edge mapping differs: '+str(y))
        independent_edges[y]=dict(available=known,notes=expected['evidence_known_ns'],status=b['status'],before=rid(y-1),after=rid(y))
    meta=load('outputs/monthly_local_linear70/META.json'); actual=load(BASE+'ROWS.json')
    require(len(meta['dates'])==len(meta['symbols'])==122 and meta['dates']==sorted(set(meta['dates'])),'Month axis')
    require(len(actual)==5978 and len(meta['etfs'])==17,'Candidate count')
    observations=[]; statuses=Counter(); kinds=Counter()
    status_names=dict(valid='checked_reported_total_chain_with_declared_source_scope',missing='missing_required_source',blocked='blocked_known_basis_break')
    for i,day in enumerate(meta['dates']):
        symbols=meta['symbols'][i]; require(len(symbols)==len(set(symbols))==49,'Candidate uniqueness')
        cutoff=closes[day]; require(str(cutoff.tz_convert('America/New_York').date())==day,'Signal date')
        for j,symbol in enumerate(symbols):
            row=actual[49*i+j]; kinds['etf' if symbol in meta['etfs'] else 'stock']+=1
            expected=dict(candidate_index=j,signal_index=i,slot_id='%03d:%02d'%(i,j),symbol=symbol,decision_date=day,
                          decision_cutoff=cutoff.isoformat(),decision_cutoff_ns=cutoff.value,policy_id=POLICY,
                          factor_admitted=False,strategy_admitted=False,factor_computed=False)
            if symbol!='MSFT':
                expected.update(status='not_applicable_etf' if symbol in meta['etfs'] else 'outside_msft_source_contract',
                                reasons=['No chain evaluated; original candidate remains on the research axis.'])
            else:
                ref=reference_choice(normalized,independent_edges,cutoff.value,day)
                require(ref['state'] in status_names,'Unexpected unresolved actual source state')
                report_ids=[rid(y) for y in ref['years']]; edge_ids=[eid(y) for y in ref['traversed']]
                expected.update(status=status_names[ref['state']],latest_report_id=rid(ref['latest']),report_ids=report_ids,edge_ids=edge_ids,
                    reasons=[] if ref['reason'] is None else [ref['reason']],
                    source_refs=[source_by_year[y]['source']['raw'] for y in ref['years']],
                    per_report_known_by={rid(y):by_id[rid(y)]['known_by'] for y in ref['years']},
                    per_edge_known_by={eid(y):edge_lookup[eid(y)]['known_by'] for y in ref['traversed']},
                    archival_vintage_assumption='original_dated_releases_retrieved_later_not_certified_first_public_bytes')
                observations.append(dict(slot_id=expected['slot_id'],decision_date=day,decision_cutoff=cutoff.isoformat(),
                    independently_selected_latest_year=ref['latest'],four_required_years=list(range(ref['latest']-3,ref['latest']+1)),
                    verified_report_ids=report_ids,verified_edge_ids=edge_ids,status=expected['status'],reason=ref['reason']))
            require(row==expected,'Actual row differs at slot '+expected['slot_id'])
            statuses[row['status']]+=1
    require(len(observations)==122 and kinds==dict(stock=3904,etf=2074),'Final candidate axis count')
    summary=load(BASE+'SUMMARY.json'); msft=Counter(x['status'] for x in observations)
    require(summary['rows']==len(actual) and summary['stock_rows']==kinds['stock'] and summary['etf_rows']==kinds['etf'],'Summary axis')
    require(summary['full_status_counts']==dict(statuses) and summary['msft_status_counts']==dict(msft),'Summary status counts')
    require(all(summary[k]==0 for k in ('new_trials','growth_ratios_computed','labels_models_ranks_returns_computed')),'Non-source work claimed')
    require(summary['factor_admitted'] is False and summary['independent_actual_chain_review_complete'] is False,'Producer overclaims admission')
    periods=[]
    for x in observations:
        if not periods or periods[-1]['status']!=x['status']:
            periods.append(dict(status=x['status'],first_decision=x['decision_date'],last_decision=x['decision_date'],count=1))
        else:
            periods[-1]['last_decision']=x['decision_date']; periods[-1]['count']+=1
    bind(str(Path(__file__).relative_to(ROOT)),digest(__file__))
    for p,h in pins.items(): require(digest(p)==h,'Input changed during independent review: '+p)
    return dict(schema_version='msft-revenue-chain-output-independent/v1',passed=True,
        input_hashes=pins,source_contract=POLICY,scope='Source-only row reconstruction from raw-source proposal and reviewed accounting boundaries.',
        independence=dict(producer_driver_imported=False,producer_evaluate_imported_or_called=False,
            calendar='Direct pinned XNYS dependency with official closure removals; full frozen session-axis equality.',
            synthetic_tests=tests_run,original_reports_checked=12,accounting_edges_checked=11,msft_cutoffs_reconstructed=122,
            complete_candidate_rows_compared=5978,stock_rows=3904,etf_rows=2074,
            growth_ratios_computed=0,price_or_return_reads=0,model_or_factor_calls=0,network_requests=0),
        status_counts=dict(msft),contiguous_status_periods=periods,report_clocks=report_checks,decisions=observations,
        admission=dict(actual_prepared_source_rows_independently_verified=True,financial_factor_admitted=False,strategy_admitted=False,
            other_issuers_admitted=False,all_event_calendar_complete=False,historical_first_public_bytes_certified=False),
        limitations=['Only bound MSFT original FY2015-FY2026 releases and reported nominal consolidated GAAP totals.',
            'FY2017-to-FY2018 accounting transition remains blocked; no later comparative replaces original data.',
            'Original-dated releases were retrieved later; no contemporaneous first-public-byte or exhaustive-revision claim.',
            'All non-MSFT candidates remain present but are outside this issuer-specific source contract.'],
        old_sources_or_observations_modified=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--self-test',action='store_true'); parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    stream=StringIO(); result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentChecks))
    require(result.wasSuccessful(),stream.getvalue())
    if args.self_test:
        print(json.dumps(dict(passed=True,synthetic_tests=result.testsRun))); return
    review=build_review(result.testsRun)
    if args.verify:
        require(load(OUTPUT)==review,'Saved review differs')
    else:
        with path(OUTPUT).open('x') as f: json.dump(review,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False); f.write('\n')
    print(json.dumps(dict(passed=True,synthetic_tests=result.testsRun,status_counts=review['status_counts'],
                         reviewed_rows=5978,review_sha256=digest(OUTPUT))))


if __name__=='__main__':
    main()
