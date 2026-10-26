"""Source-only same-report AAPL comparisons; never compute scores or returns."""
import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys
import after141_aapl_source_extract as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_aapl_comparative_source_v1'
OLD = 'outputs/monthly_profitability_extension_sources_v1/'
INPUTS = {
    OLD+'RAW_FACTS.json': '4e88dce4be908a6059481ca383d868422ae0698811b235963e260027617777c6',
    OLD+'MANIFEST.json': '4b6ecde642bbec8d7eedf6341aee97877fc9592a471d9777631e233a91f5f958',
    OLD+'EXTRACTION_PLAN.json': 'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50',
    'work/after142_profitability_admission_gaps.json': '1ac5a2fef5e586de5f98e56c5b92d02efe258479fbe999719258cd7485716405',
}
CALENDAR = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'

def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(ok, msg):
    if not ok: raise ValueError(msg)
def read_bound(rel, expected):
    p=ROOT/rel
    require(not Path(rel).is_absolute() and '..' not in Path(rel).parts and p.resolve()==p and not p.is_symlink(), 'local regular source only')
    raw=p.read_bytes(); require(sha(raw)==expected, 'digest mismatch: '+rel); return raw

def prior_cell(current, index, prior_end):
    row=deepcopy(current)
    require(index<len(row['all_row_cells']), 'missing comparison column')
    require(row['header_dates'][index]==prior_end, 'comparison date mismatch')
    row['displayed_amount']=row['all_row_cells'][index]
    row['amount_decimal']=original.decimal_cell(row['displayed_amount'])
    row['date']=prior_end
    row['locator']={'page_1based':current['locator']['page_1based'],
                    'line_1based':current['locator']['line_1based'],
                    'financial_column_index_0based':index}
    return row

def parse_comparisons(texts, spec):
    x=original.parse_layout_pages(texts,spec)
    end=x['annual_period']['prior_year_end_explicit']
    x['comparison_gross_amount']=prior_cell(x['gross_amount'],3,end)
    x['comparison_total_assets']=prior_cell(x['total_assets'],1,end)
    require(Decimal(x['comparison_total_assets']['amount_decimal'])>0, 'nonpositive comparison assets')
    x['comparison_annual_period']={'end':end,'duration_literal':'Twelve Months Ended',
        'start':None,'days_inclusive':None,
        'reason':'Prior-prior fiscal end not printed in these two source pages; do not assume 365 days.'}
    x['comparison_vintage']='same current release; not an independently observable historical event'
    x['fact_admitted']=False; x['strategy_admitted']=False
    return x

def available(release,sessions):
    date.fromisoformat(release)
    require(sessions==sorted(set(sessions)) and sessions[0]<=release<sessions[-1], 'calendar invalid or out of bounds')
    return sessions[bisect_right(sessions,release)]

def write_new(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')

def make_plan():
    OUT.mkdir(exist_ok=True)
    binds=dict(INPUTS)
    for rel,d in binds.items(): read_bound(rel,d)
    old=json.loads(read_bound(OLD+'EXTRACTION_PLAN.json',INPUTS[OLD+'EXTRACTION_PLAN.json']))
    for spec in old['sources']:
        if spec['symbol']=='AAPL':
            binds[spec['raw_path']]=spec['raw_sha256']
            for key in ['receipt','parent_release','parent_receipt']:
                binds[spec[key]['path']]=spec[key]['sha256']
    for rel in [CALENDAR,'work/after141_aapl_source_extract.py','work/after142_aapl_comparative_source.py','work/after142_aapl_comparative_tests.py']:
        binds[rel]=sha((ROOT/rel).read_bytes())
    for rel,d in binds.items(): read_bound(rel,d)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(), 'kind':'source_only_comparative_extraction',
        'symbols':['AAPL'],'fiscal_years':list(range(2016,2026)), 'input_hashes':binds,
        'all_four_cells_available':'first session strictly after CURRENT release date; never prior fiscal-end backfill',
        'prior_annual_start':'unknown unless independently established, never default365',
        'identity':'pending separate independent review','currency':'literal dollar, ISO not claimed',
        'first_public_version_verified':False,'factor_score_order_return_calls':0,'observer_writes':0}
    write_new(OUT/'PLAN.json',plan)
    print(json.dumps({'plan':str(OUT/'PLAN.json'),'sha256':sha((OUT/'PLAN.json').read_bytes())}))

def extract(plan_sha):
    planraw=(OUT/'PLAN.json').read_bytes();require(sha(planraw)==plan_sha,'plan digest mismatch')
    plan=json.loads(planraw)
    for rel,d in plan['input_hashes'].items():read_bound(rel,d)
    old=json.loads(read_bound(OLD+'EXTRACTION_PLAN.json',INPUTS[OLD+'EXTRACTION_PLAN.json']))
    oldfacts=json.loads(read_bound(OLD+'RAW_FACTS.json',INPUTS[OLD+'RAW_FACTS.json']))
    byyear={r['fiscal_year']:r for r in oldfacts['rows'] if r['symbol']=='AAPL'}
    for p,d in old['pdf_runtime_hashes'].items(): require(sha(Path(p).read_bytes())==d,'PDF runtime changed')
    runtime=old['pdf_runtime']['bundle_site_packages']
    if runtime not in sys.path:sys.path.append(runtime)
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    sessions=json.loads((ROOT/CALENDAR).read_text())['sessions']
    rows=[]
    for spec in old['sources']:
        if spec['symbol']!='AAPL':continue
        raw=read_bound(spec['raw_path'],spec['raw_sha256'])
        reader=PdfReader(io.BytesIO(raw))
        texts=[reader.pages[i].extract_text(extraction_mode='layout') for i in range(2)]
        x=parse_comparisons(texts,spec); previous=byyear[spec['fiscal_year']]
        for field in ['gross_amount','total_assets']:
            require(x[field]=={k:v for k,v in previous[field].items() if k not in ['independent_PDFium_cell_text_match','pdfium_all_row_cells']} or x[field]['amount_decimal']==previous[field]['amount_decimal'], 'current fact mismatch')
        doc=pdfium.PdfDocument(raw)
        for field,page in [('gross_amount',0),('comparison_gross_amount',0),('total_assets',1),('comparison_total_assets',1)]:
            obj=doc[page];tp=obj.get_textpage(); original.attach_geometry(x[field],tp); tp.close();obj.close()
        doc.close()
        x.update(source_path=spec['raw_path'],source_sha256=spec['raw_sha256'],source_url=spec['source_url'],
            publication=deepcopy(previous['publication']),availability=deepcopy(previous['availability']),
            provenance=deepcopy(previous['provenance']),
            identity_status='pending_separate_independent_review',
            source_only=True,semantic_comparability_admitted=False)
        require(x['availability']['candidate_first_usable_session']==available(spec['release_date'],sessions),'available date mismatch')
        x['all_four_cells_available_session']=available(spec['release_date'],sessions)
        x['historical_comparison_event_created']=False
        # A prior ORIGINAL source supplies duration metadata only, never a replacement value.
        prior=byyear.get(spec['fiscal_year']-1)
        if prior is not None:
            require(prior['annual_period']['end']==x['comparison_annual_period']['end'],'previous period mismatch')
            require(prior['publication']['release_date']<spec['release_date'],'future duration metadata forbidden')
            x['comparison_annual_period'].update(start=prior['annual_period']['start'],days_inclusive=prior['annual_period']['days_inclusive'],
                duration_source=prior['provenance']['raw_path'],duration_source_sha256=prior['provenance']['raw_sha256'],
                reason='Earlier original statement fiscal-end boundaries; current comparison VALUES still from current release.')
            x['comparison_equals_previous_original']={field:x['comparison_'+field]['amount_decimal']==prior[field]['amount_decimal'] for field in ['gross_amount','total_assets']}
        else:x['comparison_equals_previous_original']=None
        rows.append(x)
    require(len(rows)==10,'missing planned year')
    for rel,d in plan['input_hashes'].items():read_bound(rel,d)
    tests=json.loads((OUT/'TESTS.json').read_text());require(tests['passed'] and tests['synthetic_only'],'tests missing')
    require(tests['source_sha256']==sha(Path(__file__).read_bytes()),'tests stale')
    dataset={'status':'SOURCE_CANDIDATES_ONLY_NOT_ADMITTED','plan_sha256':plan_sha,'rows':rows,
        'raw_fact_events':10,'derived_ratios_factors_orders_returns':0,'observer_writes':0,
        'unknown_prior_duration_fiscal_years':[r['fiscal_year'] for r in rows if r['comparison_annual_period']['days_inclusive'] is None],
        'limits':['No independent security identity admission in this file.','Current retrieval is not proof of unchanged first-publication vintage.','Same-report comparisons are observable only at current publication.','Accounting comparability and 52/53-week normalization remain explicit downstream decisions.']}
    write_new(OUT/'COMPARATIVE_FACTS.json',dataset)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'source_only':True,'plan_sha256':plan_sha,
        'input_hashes':plan['input_hashes'],'source_bindings_rechecked':len(plan['input_hashes']),
        'dataset_sha256':sha((OUT/'COMPARATIVE_FACTS.json').read_bytes()),'tests_sha256':sha((OUT/'TESTS.json').read_bytes()),
        'raw_events':10,'monetary_cells':40,'pdf_engines_agree':True,'new_strategy_trials':0,'fact_admissions':0}
    write_new(OUT/'RECEIPT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='input_hashes'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--extract');a=p.parse_args()
    if a.plan:make_plan()
    elif a.extract:extract(a.extract)
    else:p.error('one phase required')
