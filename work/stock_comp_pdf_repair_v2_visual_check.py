"""Post-publication exact comparison to the preexisting root visual receipt.

This verifier imports no extraction producer and does not interpret new PDF
text. It compares frozen strings, metadata and file hashes only.
"""
from datetime import datetime, timezone
from pathlib import Path
import same_report144_admission as safe

ROOT=Path(__file__).resolve().parents[1]
OUT='outputs/monthly_stock_comp_pdf_repair_v2'
BASE='outputs/monthly_stock_comp_source_v1'
SELF='work/stock_comp_pdf_repair_v2_visual_check.py'
RESULT='work/stock_comp_pdf_repair_v2_visual_check.json'
PLAN_SHA='4f87e83ed86ad6487077ff83602c1fafc409be2502c73decd8921f405fd9fc87'
MANIFEST_SHA='cabb20eb4954c478fc1b0ffd90a27036ab761bf39d16b780ba74e1f735fbc3bb'
TARGETS=('AAPL:FY2021:2021-10-28','AAPL:FY2022:2022-10-27','AAPL:FY2023:2023-11-02')


def main():
    seen={}
    def read(path,expected=None):
        sha=safe.hash_file(ROOT,path)
        if expected is not None: safe.require(sha==expected,'Changed input: '+path)
        seen[path]=sha
        return safe.read(ROOT,path,sha)
    def file(path,expected):
        safe.require(safe.hash_file(ROOT,path)==expected,'Changed evidence: '+path)
        seen[path]=expected
    plan=read(OUT+'/PLAN.json',PLAN_SHA)
    manifest=read(OUT+'/MANIFEST.json',MANIFEST_SHA)
    safe.require(manifest['plan_sha256']==PLAN_SHA,'Manifest plan mismatch')
    patch=read(OUT+'/PATCH.json',manifest['output_hashes']['PATCH.json'])
    audit=read(OUT+'/AUDIT.json',manifest['output_hashes']['AUDIT.json'])
    oldplan=read(BASE+'/PLAN.json',plan['source_plan_sha256'])
    oldmanifest=read(BASE+'/MANIFEST.json',plan['source_manifest_sha256'])
    olddata=read(BASE+'/RAW_FACTS.json',oldmanifest['output_hashes']['RAW_FACTS.json'])
    for name,sha in oldmanifest['output_hashes'].items():file(BASE+'/'+name,sha)
    for mapping in (plan['source_hashes'],plan['source_code_hashes']):
        for path,sha in mapping.items():file(path,sha)
    visual=read(BASE+'/PDF_VISUAL_REVIEW.json')
    safe.require(visual['passed_visual_identification'] is True and
                 visual['plan_sha256']==plan['source_plan_sha256'] and
                 visual['raw_sha256']==oldmanifest['output_hashes']['RAW_FACTS.json'], 'Visual baseline mismatch')
    safe.require(datetime.fromisoformat(visual['reviewed_utc'])<=datetime.fromisoformat(plan['created_utc']),
                 'Visual record must predate the new extraction plan')
    safe.require([e['event_id'] for e in patch['events']]==list(TARGETS), 'Exact target patch order')
    safe.require(patch['plan_sha256']==PLAN_SHA and audit['patch_sha256']==manifest['output_hashes']['PATCH.json'], 'Patch/audit binding')
    pm={e['event_id']:e for e in oldplan['events']}; rm={e['event_id']:e for e in olddata['events']}
    vm={e['event_id']:e for e in visual['events']}
    safe.require(len(pm)==len(rm)==44 and len(vm)==10,'Complete original census/visual set')
    findings=[]
    for row in patch['events']:
        event_id=row['event_id']; spec=pm[event_id]; saved=rm[event_id]; vis=vm[event_id]
        result=row['result']; value=result['value']; loc=spec['locator_inventory']['candidates'][0]
        safe.require(saved['stock_compensation']['value'] is None and vis['v1_candidate_present'] is False,'Only prior null cells')
        safe.require(result['gaps']==[] and value is not None,'Actual repaired value required')
        for key in ('displayed_amount','value_decimal','scale','currency_literal','currency_iso','source'):
            safe.require(value[key]==vis[key],'Visual mismatch: '+event_id+' '+key)
        safe.require(value['period']==spec['periods']['current'] and
                     value['period']['period_end']==vis['current_period_end'] and
                     value['available_from']==spec['available_from'] and value['source']==spec['source'], 'Period/availability/source mismatch')
        safe.require(value['locator']==loc and loc['page_1based']==vis['page_1based'] and loc['row_label']==vis['row_label'], 'Frozen row locator mismatch')
        safe.require(loc['financial_column_index_0based']==0 and vis['column']=='left current-year column', 'Current annual column mismatch')
        safe.require(all(result[k] is False for k in ('fact_admitted','model_admitted','strategy_admitted')) and value['fact_admitted'] is False,'Admission remains false')
        safe.require(value['pdfium_text_agrees'] is True and value['pypdf_column_reconstruction']['candidate_partitions']==1,'Independent parser agreement missing')
        for ref in (spec['source'],spec['source']['receipt'],vis['rendered_image']):file(ref['path'],ref['sha256'])
        findings.append(dict(event_id=event_id,matched=True,displayed_amount=value['displayed_amount'],
                             value_decimal=value['value_decimal'],scale=value['scale'],period_end=value['period']['period_end'],
                             available_from=value['available_from'],source_sha256=value['source']['sha256']))
    result=dict(schema_version=1,stage='post_build_exact_comparison_to_preexisting_independent_visual_record',
                passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),events_checked=3,matches=findings,
                plan_sha256=PLAN_SHA,patch_sha256=manifest['output_hashes']['PATCH.json'],
                original_v1_hashes_unchanged=True,visual_record_predates_patch_plan=True,
                new_extraction_producer_imported=False,source_code_hashes={SELF:safe.hash_file(ROOT,SELF)},
                input_hashes=seen,fact_admissions=0,model_admissions=0,strategy_admissions=0,
                financial_ratios=0,limitation='Exact comparison to root visual evidence; this checker did not perform a second image inspection or certify first-public versions.')
    safe.publish(ROOT,RESULT,result)
    print(safe.encoded(dict(passed=True,events_checked=3,path=RESULT,sha256=safe.hash_file(ROOT,RESULT))).decode())


if __name__=='__main__':main()
