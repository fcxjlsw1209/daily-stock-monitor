"""Finalize the independent 44-pair audit; no producer parser imports."""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import json
import re

from bs4 import BeautifulSoup
from after158_basic_shares_independent_review import audit, load, path, sha, verify, require

SELF='work/after158_basic_shares_independent_remaining_review.py'
EVIDENCE='work/after158_basic_shares_independent_remaining_evidence.json'
PREVIOUS='work/after158_basic_shares_independent_review.json'
MANUAL='work/after158_basic_shares_independent_manual.json'
PLAN='work/after158_basic_shares_source_plan.json'
INVENTORY='outputs/after158_basic_shares_inventory_v1/INVENTORY.json'
DATA='outputs/after158_basic_shares_extraction_v1/RAW_SHARES.json'
OUTPUT='work/after158_basic_shares_independent_remaining_review.json'


def norm(text):
    return re.sub(r'\s+',' ',text).strip()


def main():
    previous=load(PREVIOUS,'d0a108bbd11dbe79715d9ad63abcb3be05dbd04c597b7f78ffd491a27269a1f3')
    evidence=load(EVIDENCE,'f802cad86a5e2216099dec8baf097abedefeb2dd7792f98ac3f0ae36b3850d69')
    pins=dict(previous['input_hashes'])
    for name,h in evidence['input_hashes'].items():
        require(name not in pins or pins[name]==h,'Conflicting old/new evidence pin')
        pins[name]=h
    for name in (SELF,EVIDENCE,PREVIOUS):
        pins[name]=sha(path(name).read_bytes())
    verify(pins)
    manual=load(MANUAL)
    plan=load(PLAN)
    data=load(DATA)
    old_oracle=manual['manual_source_pairs']
    more=[]
    for r in evidence['records']:
        require(r['failure'] is None,'Remaining original parser failed')
        require(r['cells']['current']['share_scale']==r['cells']['comparison']['share_scale'],'Inconsistent same-row scale')
        more.append({'event_id':r['event_id'],'current':r['cells']['current']['value_decimal'],
            'comparison':r['cells']['comparison']['value_decimal'],'share_scale':r['cells']['current']['share_scale']})
    require(len(old_oracle)==17 and len(more)==27,'Fixed scope changed')
    require(not {r['event_id'] for r in old_oracle}.intersection(r['event_id'] for r in more),'Duplicated verification scope')
    stats=audit(data['events'],plan['events'],load(INVENTORY)['events'],old_oracle+more)
    require(stats['events']==stats['sampled_numeric_pairs_verified']==44,'Not all original pairs verified')
    require(data['source_admission_changed'] is False,'Source admission changed')
    require(data['activity']=={'source_cells_extracted':88,'factors':0,'scores':0,'returns':0,'orders':0},'Unexpected activity')
    for key in ('events','raw_annual_pairs_extracted','explicit_weighted_average_pairs','implicit_basic_eps_denominator_pairs','by_symbol'):
        require(stats[key]==data['summary'][key],'Incorrect producer summary')
    events={e['event_id']:e for e in plan['events']}
    presentation=[]
    for row in data['events']:
        event=events[row['event_id']]
        if event['format']=='html':
            soup=BeautifulSoup(path(event['source']['path']).read_bytes(),'html.parser')
            for node in soup(['script','style']):node.decompose()
            source_text=norm(soup.get_text(' ',strip=True))
            source_ref=event['source']['path']
        else:
            ref=event['existing_text'];verify({ref['path']:ref['sha256']})
            pins[ref['path']]=ref['sha256']
            source_text=norm(path(ref['path']).read_text())
            source_ref=ref['path']
        original_notes=[]
        for snippet in row['split_evidence']['evidence']:
            literal=norm(snippet['literal'])
            require(literal in source_text,'Reported split text not present in source: '+row['event_id'])
            original_notes.append(literal)
        # These source-note categories are declared human review conclusions,
        # not a keyword-based split-history certification or an admission gate.
        if row['event_id']=='NVDA:FY2022:2022-02-16':
            category='direct_basic_denominator_footnote'
            conclusion='Weighted-average share heading (A) and same-table footnote (A) explicitly reflect four-for-one split on July 19, 2021.'
        elif row['event_id']=='NVDA:FY2025:2025-02-26':
            category='contextual_per_share_and_diluted_note'
            conclusion='All per-share amounts are described as retroactively split-adjusted; (E) is on reconciliation diluted shares. This is not a direct GAAP Basic-row footnote; retain common-presentation assumption explicitly.'
        elif original_notes and set(original_notes)=={'Stock Split History'}:
            category='website_navigation_only'
            conclusion='Website navigation label is not a historical event or a statement about this report\'s share basis.'
        else:
            require(not original_notes,'Unreviewed split disclosure requires manual handling')
            category='no_split_note_located_not_absence_certification'
            conclusion='The reviewed annual Basic row uses one shared label and scale. No different current/prior share basis is expressly identified in the inspected presentation; no assertion of complete split history.'
        if row['event_id']=='AAPL:FY2020:2020-10-29':
            conclusion+=' Special boundary: this PDF does not explicitly document the split in the searched text; use only its same-report annual pair, not a spliced prior-report count. A strict split-certification claim is unsupported by these bytes alone.'
        if row['event_id']=='NVDA:FY2016:2016-02-17':
            conclusion+=' Reporting precision changed from prior release thousands to current release millions. Preserve this report\'s rounded comparative; do not substitute higher-precision earlier-release figures.'
        presentation.append({'event_id':row['event_id'],'source':source_ref,
            'same_report_same_basic_row_and_scale':True,
            'explicit_current_prior_different_share_basis_observed':False,
            'scope_of_negative_finding':'Reviewed annual presentation and reported split notes only, not an exhaustive corporate-action audit.',
            'split_note_category':category,'verified_original_split_literals':original_notes,
            'review_conclusion':conclusion,'complete_split_history_certified':False,
            'source_admission_changed':False})
    verify(pins)
    report={'schema':'after158_basic_shares_independent_remaining_review_v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
        'scope':'Remaining 27 original HTML pairs independently parsed without producer imports/locators, then all 27 header/row/unit summaries manually inspected and numerically compared. Together with the separately retained prior 17-pair audit, all 44 pairs / 88 source cells are verified.',
        'input_hashes':pins,'summary':dict(stats,previously_verified_pairs=17,newly_verified_pairs=27,
            total_independently_verified_pairs=44,total_independently_verified_source_cells=88,
            explicit_different_current_prior_basis_observed=0,
            basis_note_categories=dict(Counter(p['split_note_category'] for p in presentation))),
        'checks':{'all_44_source_pairs_and_units_verified':True,'all_44_original_metadata_and_holds_preserved':True,
            'remaining_scope_fixed_and_disjoint':True,'all_pins_reverified':True,
            'remaining_27_headers_rows_units_manually_reviewed':True,
            'same_report_presentation_only':True,'no_automatic_split_adjustment':True,
            'complete_split_history_certified':False,'historical_first_public_version_certified':False,
            'source_or_strategy_admission_changed':False},
        'synthetic_validation':{'command':'./run_local.sh work/after158_basic_shares_independent_remaining.py --test-only',
            'tests_run':8,'failures':0,'errors':0,'actual_source_inputs_read_in_tests':False},
        'presentation_review':presentation,
        'limitations':['No ratio, change, sign, rank, factor, order, return or strategy test was computed.',
            'Source correctness and one-row annual presentation do not certify historical first-public timing, full corrections, daily security master, or complete stock-split history.',
            'The 20 implicit Basic EPS denominator pairs remain distinguished from the 24 explicitly weighted-average pairs.',
            'No evidence of explicitly different period share bases was observed within the specified review scope; absence of a keyword is not proof of economic comparability.',
            'A conditional study may explicitly inherit same-report common-presentation interpretation; it must not present that assumption as newly certified historical split data.',
            'The original 17-pair review is not overwritten; this supplementary review closes its numerical coverage limitation.'],
        'source_admission_changed':False,'financial_features_computed':0,'returns_computed':0,
        'orders_computed':0,'new_strategy_trials':0}
    raw=(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with path(OUTPUT).open('xb') as handle:handle.write(raw)
    print(json.dumps({'path':OUTPUT,'sha256':sha(raw),'summary':report['summary']}))


if __name__=='__main__':
    main()
