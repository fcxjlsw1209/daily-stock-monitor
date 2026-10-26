"""Conditionally admit pinned issuer releases for limited offline research only.
No factor evaluation, ranking, strategy execution, or return computation.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
REVIEW = 'work/profitability141_issuer_statements_independent_review.json'
REVIEW_SHA = 'dfb8146993a727d5535108a5908ad82ed8d24bafa420add7f544b7a2258a7af5'
PROPOSAL_SHA = '80901aab35840451f289e58ca90caefefa751c717561e8cb1175d39012fd7a58'
META = 'outputs/monthly_local_linear70/META.json'
OUT = 'outputs/monthly_profitability141_issuer_admission'
SELF = 'work/profitability141_issuer_source_admission.py'
TEST = 'work/test_profitability141_issuer_source_admission.py'
ISSUERS = {'MSFT':'0000789019','NVDA':'0001045810'}
POLICY = {
    'scope':'Limited-coverage exploratory historical-disclosure research; only MSFT and NVDA issuer releases.',
    'first_public_version_verified':False,
    'historical_content_assumption':'The current-retrieval official issuer page preserves the originally published current-annual statement values. This is an explicit unverified version-continuity assumption, not first-public-version or archive-vintage certification.',
    'identity_rule':'Each event uses only its own original dated issuer/ticker anchor. Identity validity begins at that event next-session availability and ends at its period_end plus 550 calendar days. This is a fixed causal continuity assumption, not certification of a daily security master. Later releases never determine earlier validity intervals.',
    'calendar':'Pinned XNYS exchange_calendars plus official closure overlay; first session strictly later than issuer-local publication date, not individually exchange-certified daily history.',
    'amounts':'Preserve exact displayed Decimal strings and separate scales; no ratios or normalization are computed here. Same-issuer dollar reporting labels are assumed stable across annual releases. ISO USD unresolved remains unresolved.',
    'semantics':'MSFT monetary Gross margin and NVDA Gross profit in the original annual statement; same-release year-end total assets. This is issuer_statement evidence, not SEC us-gaap context certification.',
    'period':'Start derived as one day after the same balance sheet previous year-end; annual durations independently checked and explicitly not presented as verbatim start dates.',
    'event_selection':'All registered annual events are retained, including invalid events. Downstream selects latest known annual event before validity filtering; a newer invalid event blocks old-value fallback.',
    'catalog_limit':'All 12 Q4 events per issuer FY2015-FY2026 in the pinned catalogs are covered; omitted corrections or reissues outside those catalogs are not certified absent.',
    'missing':'Every original META security remains present. All other issuers/ETFs have empty events and receive the registered neutral fallback.',
    'source_stop':'No network requests. Existing SEC403 stop for 2026-10-19 America/Los_Angeles is unchanged; SEC facts remain unadmitted.'
}


def sha(raw): return hashlib.sha256(raw).hexdigest()

def canonical_day(value):
    if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:
        raise ValueError('Canonical date required')
    return date.fromisoformat(value)

def norm(text): return re.sub(r'\s+',' ',text).strip()

def decimal_string(value, positive=False):
    if not isinstance(value,str) or not value or value.strip()!=value:
        raise ValueError('Exact decimal string required')
    try: amount=Decimal(value)
    except InvalidOperation: raise ValueError('Invalid decimal')
    if not amount.is_finite() or (positive and amount<=0):
        raise ValueError('Invalid finite positive decimal')
    return value

def local(relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts: raise ValueError('Local relative path required')
    p=ROOT/p
    if p.resolve()!=p: raise ValueError('Symlinks prohibited')
    return p

def pinned(binding):
    raw=local(binding['path']).read_bytes()
    if sha(raw)!=binding['sha256']: raise ValueError('Digest mismatch: '+binding['path'])
    return raw


def make_event(raw, anchor, sessions):
    """Preserve an invalid annual event as a chronology barrier; no amounts used."""
    symbol=raw['symbol']
    if symbol not in ISSUERS: raise ValueError('Unregistered issuer')
    availability=raw['availability']
    release=canonical_day(availability['release_date'])
    if not sessions or sessions!=sorted(set(sessions)):
        raise ValueError('Invalid session sequence')
    eligible=[d for d in sessions if canonical_day(d)>release]
    if not eligible: raise ValueError('Publication outside calendar')
    available=eligible[0]
    period=raw['annual_period']
    end=canonical_day(period['end']);start=canonical_day(period['start'])
    through=(end+timedelta(days=550)).isoformat()
    event={
        'event_id':'issuer_release:'+symbol+':FY'+str(raw['fiscal_year_label'])+':'+raw['source']['raw_sha256'][:16],
        'available_from':available,'publication_date':release.isoformat(),
        'admitted':False,'original_annual':True,'measurement_eligible':True,
        'issuer_id':'CIK'+ISSUERS[symbol],'security_id':'research_equity:'+symbol,
        'currency_key':'issuer_reporting_dollar:'+symbol,
        'identity_valid_from':available,'identity_valid_through':through,
        'period_start':start.isoformat(),'period_end':end.isoformat(),
        'fiscal_year_label':raw['fiscal_year_label'],
        'first_public_version_verified':False,
        'admission_scope':'conditional_current_retrieval_issuer_release_research',
        'identity_basis':'same_event_dated_issuer_ticker_anchor_and_fixed_550_day_continuity_assumption',
        'currency_iso_verified':raw['currency']['USD_verified'],
        'currency_iso':raw['currency']['iso_currency'],
        'annual_start_is_verbatim':False,'source':raw['source'],
        'original_raw_fact_pair':raw,'identity_anchor':anchor,
        'admission_rejections':[]
    }
    try:
        if availability['first_conservative_session_after_release_date']!=available:
            raise ValueError('Stored availability differs from pinned calendar')
        if raw['cik']!=ISSUERS[symbol]: raise ValueError('Issuer CIK mismatch')
        if anchor is None: raise ValueError('Missing same-event identity anchor')
        if (anchor['symbol'],anchor['fiscal_year_label'],anchor['release_date'])!=(symbol,raw['fiscal_year_label'],release.isoformat()):
            raise ValueError('Identity anchor event mismatch')
        if anchor['raw']['sha256']!=raw['source']['raw_sha256'] or anchor['source_url']!=raw['source']['source_url']:
            raise ValueError('Identity anchor raw source mismatch')
        if not isinstance(anchor.get('issuer_ticker_literal'),str) or not anchor['issuer_ticker_literal'].strip():
            raise ValueError('Original ticker literal missing')
        if not start<=end<release or not available<=through:
            raise ValueError('Noncausal annual date bounds')
        if not 330<=(end-start).days+1<=400: raise ValueError('Nonannual duration')
        if period['days_inclusive']!=(end-start).days+1: raise ValueError('Duration mismatch')
        if raw['gaap']['selected_formal_consolidated_statements'] is not True or raw['gaap']['non_gaap_tables_excluded'] is not True:
            raise ValueError('Unverified annual statement semantics')
        if raw['source']['receipt_and_event_verified'] is not True or raw['raw_fact_pair_extracted'] is not True:
            raise ValueError('Unverified raw source')
        expected='Gross margin' if symbol=='MSFT' else 'Gross profit'
        gp=raw['gross_amount'];assets=raw['total_assets']
        if gp['measure_label']!=expected or assets['measure_label']!='Total assets':
            raise ValueError('Wrong financial measures')
        if gp['date']!=end.isoformat() or assets['date']!=end.isoformat():
            raise ValueError('Mismatched annual end')
        if gp['currency_label']!='$' or assets['currency_label']!='$':
            raise ValueError('Unresolved same-report currency')
        if raw['currency']['both_statement_currency_labels_match'] is not True:
            raise ValueError('Currency match absent')
        for row in [gp,assets]:
            if type(row['scale']) is not int or row['scale'] not in [1000,1000000]:
                raise ValueError('Unregistered scale')
        event['gross_profit']={'value':decimal_string(gp['amount_decimal']),'scale':str(gp['scale'])}
        event['assets']={'value':decimal_string(assets['amount_decimal'],positive=True),'scale':str(assets['scale'])}
        event['admitted']=True
    except (KeyError,TypeError,ValueError) as exc:
        event['admission_rejections'].append(str(exc))
    return event


def build_dataset(pairs, anchors, sessions, symbols):
    if len(symbols)!=len(set(symbols)): raise ValueError('Duplicate original symbol')
    events={symbol:[] for symbol in symbols}
    anchor_map={}
    for anchor in anchors:
        k=(anchor['symbol'],anchor['fiscal_year_label'])
        if k in anchor_map: raise ValueError('Duplicate identity anchor')
        anchor_map[k]=anchor
    seen=set()
    for raw in pairs:
        k=(raw['symbol'],raw['fiscal_year_label'])
        if k in seen: raise ValueError('Duplicate annual event')
        seen.add(k)
        if raw['symbol'] not in events: raise ValueError('Issuer absent from original universe')
        events[raw['symbol']].append(make_event(raw,anchor_map.get(k),sessions))
    for rows in events.values(): rows.sort(key=lambda r:(r['available_from'],r['event_id']))
    return {'schema_version':1,'events':events,'admission_policy':POLICY,
            'universe_symbols':list(symbols),'source_version_class':'current_retrieval_official_issuer_pages',
            'first_public_version_verified':False,'no_factor_computations':True}


def execute(proposal_path):
    review=json.loads(pinned({'path':REVIEW,'sha256':REVIEW_SHA}))
    if review['status']!='PASS_EXISTING_RAW_SOURCE_CONTRACT_WITH_EXPLICIT_RESEARCH_LIMITATIONS':
        raise ValueError('Independent source review failed')
    bindings=review['bindings'];raw=json.loads(pinned(bindings['existing_raw_facts']))
    plan=json.loads(pinned(bindings['plan']))
    source_hashes={REVIEW:REVIEW_SHA}
    def bind(binding):
        data=pinned(binding);source_hashes[binding['path']]=binding['sha256'];return data
    for name in ['existing_raw_facts','plan','extractor','extractor_tests','review_script']:bind(bindings[name])
    inputs={name:json.loads(bind(binding)) for name,binding in bindings['inputs'].items()}
    bind(plan['separate_from_unchanged_SEC_DATA_PLAN'])
    for path,digest in inputs['calendar_plan']['bindings'].items():bind({'path':path,'sha256':digest})
    for spec in plan['source_scope']['sources']:
        bind({'path':spec['raw_path'],'sha256':spec['raw_sha256']})
        bind({'path':spec['receipt_path'],'sha256':spec['receipt_sha256']})
    meta_raw=local(META).read_bytes();meta=json.loads(meta_raw);source_hashes[META]=sha(meta_raw)
    symbols=sorted({s for row in meta['symbols'] for s in row})
    etfs=meta['etfs']
    stocks=sorted(set(symbols)-set(etfs))
    if len(symbols)!=90 or len(etfs)!=17 or len(set(etfs))!=17 or len(stocks)!=73: raise ValueError('Original73 stocks plus17 ETFs mismatch')
    if len(meta['symbols'])!=122 or any(len(row)!=49 or len(set(row))!=49 for row in meta['symbols']): raise ValueError('Original122 by49 axes mismatch')
    identity=inputs['identity_review']
    anchors=identity['focused_original_identity_evidence']['annual_issuer_ticker_observations']
    pair_keys={(p['symbol'],p['fiscal_year_label']) for p in raw['pairs']}
    expected={(s,y) for s in ISSUERS for y in range(2015,2027)}
    annual_events=[e for e in inputs['MSFT_events']+inputs['NVDA_events'] if e['fiscal_quarter']==4]
    if pair_keys!=expected or len(raw['pairs'])!=24 or {(e['ticker'],e['fiscal_year']) for e in annual_events}!=expected or len(annual_events)!=24:
        raise ValueError('Registered annual event coverage mismatch')
    for anchor in anchors:
        raw_page=bind(anchor['raw'])
        text=norm(BeautifulSoup(raw_page,'html.parser').get_text(' ',strip=True))
        if norm(anchor['issuer_ticker_literal']) not in text: raise ValueError('Dated issuer/ticker literal absent')
    dataset=build_dataset(raw['pairs'],anchors,inputs['calendar_sessions']['sessions'],symbols)
    flattened=[e for rows in dataset['events'].values() for e in rows]
    if len(flattened)!=24 or not all(e['admitted'] for e in flattened): raise ValueError('Registered source admission failed')
    proposal_raw=local(proposal_path).read_bytes();proposal=json.loads(proposal_raw)
    if sha(proposal_raw)!=PROPOSAL_SHA: raise ValueError('Final proposal digest mismatch')
    if proposal.get('round',proposal.get('prospective_round'))!=141: raise ValueError('Wrong prospective proposal round')
    source_hashes[proposal_path]=sha(proposal_raw)
    for rel in [SELF,TEST,'local_run.py','run_local.sh','local.sandbox.sb']:source_hashes[rel]=sha(local(rel).read_bytes())
    test_results=unittest_result()
    output=local(OUT)
    if output.exists(): raise ValueError('Admission output already exists; never overwrite')
    output.mkdir()
    def save(name,value):
        data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        with (output/name).open('xb') as f:f.write(data)
        return sha(data)
    policy_sha=save('ADMISSION_POLICY.json',POLICY)
    dataset_sha=save('DATASET.json',dataset)
    source_hashes[OUT+'/ADMISSION_POLICY.json']=policy_sha
    receipt={
        'schema_version':1,'round':141,'stage':'source_admission','passed':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_path':OUT+'/DATASET.json','dataset_sha256':dataset_sha,
        'proposal_sha256':sha(proposal_raw),'meta_sha256':sha(meta_raw),
        'sessions_sha256':bindings['inputs']['calendar_sessions']['sha256'],'source_hashes':source_hashes,
        'semantic_review_passed':True,'availability_review_passed':True,'identity_review_passed':True,'period_review_passed':True,
        'review_passes_are_conditional':True,'conditions':POLICY,
        'admitted_events':24,'admitted_symbols':['MSFT','NVDA'],'original_universe_count':len(symbols),'original_stock_count':len(stocks),'original_etf_count':len(etfs),
        'first_public_version_verified':False,'SEC_companyfacts_admitted':0,
        'factor_or_return_calculations':0,'network_requests':0,
        'tests':test_results,'independent_review':{'path':REVIEW,'sha256':REVIEW_SHA},
        'identity_intervals_depend_on_future_events':False,
        'event_availability':[{k:e[k] for k in ['event_id','available_from','period_start','period_end','identity_valid_from','identity_valid_through']} for e in flattened]
    }
    receipt_sha=save('RECEIPT.json',receipt)
    print(json.dumps({'dataset_path':OUT+'/DATASET.json','dataset_sha256':dataset_sha,'receipt_path':OUT+'/RECEIPT.json','receipt_sha256':receipt_sha,'admitted_events':24,'original_symbols':len(symbols),'factor_or_return_calculations':0,'tests':test_results}))


def unittest_result():
    import io
    import unittest
    import test_profitability141_issuer_source_admission as tests
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():raise ValueError(stream.getvalue())
    return {'passed':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'output':stream.getvalue()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--proposal',required=True);args=parser.parse_args();execute(args.proposal)
