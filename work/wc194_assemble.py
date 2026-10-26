"""194 write-once source admission. No W sum, ratios, features or returns."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
import wc194_admission as gate
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_working_capital194_source/admitted'
SCOPE='work/working_capital194_source_scope.json';SCOPE_SHA='f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be'
CONCEPT='work/working_capital194_experiment_concept.json';CONCEPT_SHA='380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83'
CALENDAR='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json';CALENDAR_SHA='c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'
RUNTIME='work/capex193_pdf_runtime/RUNTIME.json';RUNTIME_SHA='ad437674d06761ef278da476110c1d3aedf386cd5a0276161304627d2140c98a'
PARTS={'outputs/monthly_working_capital194_source/msft_aapl/EXTRACTION.json':'6ff8c01f9fc3cffb6f47a3f7ceb4792d3647208f008aa1189139565292c5c727','outputs/monthly_working_capital194_source/nvda_amd/EXTRACTION.json':'7353e0ab69414a55b16fada6e91086c2f6f661624739b1a035fe381e985e455b'}
VISUAL='outputs/monthly_working_capital194_source/msft_aapl/VISUAL_REVIEW.json';VISUAL_SHA='3cea8804b91ce7aeb4db8226820213c0fa0278f0f7dffa7c1547105f743095c7'
REVIEW='outputs/monthly_working_capital194_source/independent/REVIEW.json'
require=gate.require
MAPS=('input_sha256','source_hashes','code_sha256','code_hashes','pdf_runtime_hashes','image_hashes','producer_refs','producer_visual_refs','expected_ref','tests_ref','producer_code_sha256','preflight_ref')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encoded(obj):return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def local(p):
    path=Path(p);path=path if path.is_absolute() else ROOT/path
    require(path.resolve()==path and path.is_relative_to(ROOT),'Regular research path required')
    return path

def validate_reviews(review,visual,original):
    require(review.get('passed') is True and review.get('source_only') is True,'Independent source review incomplete')
    require(review.get('scope_sha256')==SCOPE_SHA and review.get('concept_sha256')==CONCEPT_SHA,'Independent scope/concept differs')
    require(review.get('producer_refs')==PARTS and review.get('producer_visual_refs')=={VISUAL:VISUAL_SHA},'Independent producer binding differs')
    require(review.get('runtime',{}).get('manifest_sha256')==RUNTIME_SHA and review['runtime'].get('historical_runtime_currently_passes') is False,'New isolated runtime required')
    require(review.get('component_sum_computed') is False and all(review.get(k) is False for k in ('ratios_computed','factors_computed','orders_computed','returns_computed')),'Source review exceeds source-only scope')
    require(visual.get('passed') is True and visual.get('visual_review_pending') is False and visual.get('report_count')==10 and visual.get('current_cells_reviewed')==30,'Complete ten-PDF visual check required')
    require(visual.get('extraction_sha256')==PARTS[visual.get('extraction_path')],'Visual extraction differs')
    require(all(type(visual.get(k)) is int and visual[k]==0 for k in ('component_sums','financial_ratios','factors','orders','returns','fact_admissions','strategy_admissions')),'Visual review exceeds source-only scope')
    apple=[e['event_id'] for e in original['events'] if e['symbol']=='AAPL']
    rows=gate.indexed(visual['records'],apple)
    require(len(apple)==10,'Exactly ten original AAPL PDFs required')
    for row in rows.values():
        require(all(row.get(k) is True for k in ('pure_trade_ar_separate_from_vendor_nontrade','inventories_and_ap_distinct_from_other_assets_and_liabilities','all_three_rows_within_operating_cashflow_changes','legible_unclipped_annual_columns','producer_visual_review_completed')),'Incomplete producer visual check')
    independent=gate.indexed(review.get('independent_visual_reviews'),apple)
    require(all(r.get('visual_source')=='independent_generated_PDF_page_directly_viewed' and all(r.get(k) is True for k in ('units_millions','three_distinct_operating_reconciliation_rows','signs_and_values_match_independent_extraction')) and r.get('current_annual_column')==0 for r in independent.values()),'Independent PDF visual check incomplete')
    for eid,r in independent.items():require(r.get('source_sha256')==rows[eid]['source']['sha256'],'Independent visual source mismatch')
    ids=[e['event_id'] for e in original['events']]
    ev=gate.indexed(review['events'],ids)
    require(all(r.get('review_passed') is True and r.get('producer_comparison_passed') is True for r in ev.values()),'Incomplete44 event comparison')

def main(review_sha):
    require(not OUT.exists(),'Write-once admission already exists')
    pins={}
    def bind(p,h):
        f=local(p);require(sha(f)==h,'Closure SHA mismatch: '+str(p));key=str(f.relative_to(ROOT))
        require(key not in pins or pins[key]==h,'Conflicting SHA');pins[key]=h
    def read(p,h):bind(p,h);return json.loads(local(p).read_text())
    def bind_object(obj):
        for group in MAPS:
            mapping=obj.get(group,{})
            require(type(mapping) is dict,'Expected SHA mapping: '+group)
            for p,h in mapping.items():bind(p,h)
    scope=read(SCOPE,SCOPE_SHA);concept=read(CONCEPT,CONCEPT_SHA)
    require(concept['registered'] is False and all(concept[k]==0 for k in ('actual_new_factors','actual_new_orders','actual_new_returns')),'Numerical experiment already started')
    require(scope['component_sum_permitted'] is False,'Source scope permits sums')
    old={p:read(p,h) for p,h in scope['input_sha256'].items()}
    original=old['outputs/monthly_cash_quality_source_admission_v1/DATASET.json'];ids=[e['event_id'] for e in original['events']]
    require(len(ids)==44 and sum(e['cash_quality_admission']['admitted'] for e in original['events'])==31,'Original44/31/13 source census differs')
    sessions=read(CALENDAR,CALENDAR_SHA)['sessions'];rt=read(RUNTIME,RUNTIME_SHA)
    for p,h in rt['pdf_runtime_hashes'].items():bind(p,h)
    acquisition=read(rt['acquisition']['path'],rt['acquisition']['sha256'])
    require(acquisition['package_executed'] is False and acquisition['RECORD_verified_members']==70,'Isolated wheel acquisition evidence required')
    for t in scope['prospective_targets']:
        for ref in (t['source'],t['source']['receipt']):bind(ref['path'],ref['sha256'])
    parts=[read(p,h) for p,h in PARTS.items()];visual=read(VISUAL,VISUAL_SHA);review=read(REVIEW,review_sha)
    validate_reviews(review,visual,original)
    for obj in parts+[visual,review]:bind_object(obj)
    bind('work/wc194_msft_aapl_visual_review.py',visual['review_code_sha256'])
    expected_refs=review.get('expected_ref',{})
    require(len(expected_refs)==1,'One independent raw expected set required')
    expected=read(*next(iter(expected_refs.items())));bind_object(expected)
    require(expected['scope_sha256']==SCOPE_SHA and expected['component_sum_computed'] is False,'Raw expected source scope differs')
    target=gate.indexed(scope['prospective_targets'],ids);producer=gate.indexed([r for part in parts for r in part['records']],ids);independent=gate.indexed(review['events'],ids)
    result=[gate.check_record(e,target[e['event_id']],producer[e['event_id']],independent[e['event_id']],sessions) for e in original['events']]
    for name in ('wc194_admission.py','test_wc194_admission.py','wc194_assemble.py','test_wc194_assemble.py'):bind('work/'+name,sha(ROOT/'work'/name))
    for row in visual['records']:bind(row['render']['path'],row['render']['sha256'])
    policy=dict(schema_version=1,method=gate.METHOD,definition=gate.DEFINITION,source_scope_sha256=SCOPE_SHA,concept_sha256=CONCEPT_SHA,created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,current_report_trade_working_capital_only=True,use_comparison_flows=False,preserve_all_original_holds=True,all_three_components_required=True,component_sum_permitted=False,clock='first pinned exchange session strictly after report release date',annual_days=[330,400],max_age_days=550,same_report_literal_currency_assumption=True,conditional_research_only=True,financial_ratios_permitted=False,factor_computation_permitted=False,return_computation_permitted=False,scope_limitations=scope['source_limitations'],old_extraction_runtime_status='Historical receipt preserved; old shared PDFium current hash fails. New193 isolated runtime verifies all126 files; no old-runtime pass is claimed.',input_hashes=pins)
    ph=hashlib.sha256(encoded(policy)).hexdigest();dataset=dict(schema_version=1,method=gate.METHOD,events=result,policy_sha256=ph,conditional_research_only=True);dh=hashlib.sha256(encoded(dataset)).hexdigest()
    admitted=[r['event_id'] for r in result if r['trade_working_capital_admission']['admitted']];held=[r['event_id'] for r in result if not r['trade_working_capital_admission']['admitted']]
    receipt=dict(schema_version=1,stage='conditional_trade_working_capital_source_admission',passed=True,events=44,admitted_events=len(admitted),held_events=len(held),admitted_event_ids=admitted,held_event_ids=held,inherited_admitted_events=31,inherited_held_events=13,all_original_event_values_preserved=True,data_sha256=dh,policy_sha256=ph,independent_review=dict(path=REVIEW,sha256=review_sha),first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,component_sums=0,financial_ratios=0,factor_calculations=0,return_calculations=0,observer_changes=0)
    for p,h in pins.items():require(sha(ROOT/p)==h,'Input changed during admission: '+p)
    OUT.mkdir(parents=True)
    for name,obj in [('POLICY.json',policy),('DATASET.json',dataset),('ADMISSION.json',receipt)]:
        with (OUT/name).open('xb') as f:f.write(encoded(obj))
    print(json.dumps(dict(passed=True,events=44,admitted=len(admitted),held=len(held),data_sha256=dh,policy_sha256=ph,receipt_sha256=sha(OUT/'ADMISSION.json'),closure_pins=len(pins))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review-sha',required=True);a=p.parse_args();main(a.review_sha)
