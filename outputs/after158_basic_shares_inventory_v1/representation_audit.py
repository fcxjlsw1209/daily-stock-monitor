"""Quantity-free follow-up for the frozen inventory's flattened Apple page text."""
import argparse
from datetime import datetime,timezone
import json
import re
import unittest
import after158_basic_shares_inventory as base

SELF=base.OUT+'/representation_audit.py'
OUT=base.OUT+'/PDF_TEXT_ADDENDUM.json'
PLAN_SHA='3ff000c9a1b9c17d04d2a2edf2f79f21b18811f4e32263b38b768ace805ca60b'
INV_SHA='3620ca72d412624c5cc7f0167d9988292f5fb01f66a618d45f385dbcfd379121'
HEADING=re.compile(r'shares\s+used\s+in\s+computing\s+(?:earnings|net\s+income)\s+per\s+share\s*:',re.I)


def locate(text):
    result=[]
    for heading in HEADING.finditer(text):
        suffix=text[heading.end():heading.end()+300]
        basic=re.search(r'\bbasic\b',suffix,re.I)
        diluted=re.search(r'\bdiluted\b',suffix,re.I)
        if basic is None or(diluted is not None and diluted.start()<basic.start()):continue
        result.append(dict(heading_character_start=heading.start(),heading_character_end=heading.end(),
                           basic_character_start=heading.end()+basic.start(),
                           heading_literal=heading.group(0),basic_label=basic.group(0),
                           semantic_kind='basic_eps_denominator_weighting_implicit',
                           annual_heading_present=bool(base.ANNUAL.search(text)),
                           interim_heading_present=bool(base.INTERIM.search(text)),
                           same_report_annual_pair_locator_present=False,
                           blockers=['flattened_pdf_text_requires_annual_column_geometry_review','weighted_average_semantics_implicit_needs_review'],
                           quantities_retained=False))
    return result


class Tests(unittest.TestCase):
    def test_flattened_positive(self):
        value=locate('EPS Basic 123.45 Diluted 123.44 Shares used in computing earnings per share: Basic 87654321 Diluted 98765432')
        self.assertEqual(len(value),1);self.assertNotIn('87654321',json.dumps(value));self.assertNotIn('98765432',json.dumps(value))
    def test_eps_only(self):self.assertEqual(locate('Earnings per share: Basic 87654321 Diluted 98765432'),[])
    def test_diluted_only(self):self.assertEqual(locate('Shares used in computing earnings per share: Diluted 98765432'),[])
    def test_bound(self):self.assertEqual(locate('Shares used in computing earnings per share:'+(' '*301)+'Basic'),[])


def build():
    base.require(base.sha(base.ROOT/base.PLAN)==PLAN_SHA,'plan');plan=base.read(base.PLAN)
    base.verify(plan['source_hashes']);base.verify(plan['implementation_hashes'])
    base.require(base.sha(base.ROOT/base.OUT/'INVENTORY.json')==INV_SHA,'inventory')
    inventory=base.read(base.OUT+'/INVENTORY.json');rows=[]
    for event in plan['events']:
        if event['format']!='pdf':continue
        text=(base.ROOT/event['existing_text']['path']).read_text()
        rows.append(dict(event_id=event['event_id'],source=event['source'],existing_text=event['existing_text'],
                         flattened_text_line_count=len(text.splitlines()),candidates=locate(text),
                         lexical_review_flags=base.word_flags(text),
                         interpretation='The original inventory no-candidate status describes its line-oriented locator, not absence of a source label. Annual current/prior geometric attribution remains unverified.'))
    count=sum(bool(r['candidates'])for r in rows)
    byid={r['event_id']:r for r in rows}
    basic_count=sum(bool(r['inventory']['candidates'])or bool(byid.get(r['event_id'],{}).get('candidates'))for r in inventory['events'])
    base.verify(plan['source_hashes']);base.verify(plan['implementation_hashes'])
    result=dict(schema='after158_basic_shares_flattened_pdf_text_followup_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                source_admission_changed=False,original_plan_and_inventory_unchanged=True,
                scope='Same ten PDF texts already bound before the first scan; no new source, issuer, event, or data value.',
                input_hashes={base.PLAN:PLAN_SHA,base.OUT+'/INVENTORY.json':INV_SHA,base.OUT+'/MANIFEST.json':base.sha(base.ROOT/base.OUT/'MANIFEST.json'),SELF:base.sha(base.ROOT/SELF),**{x['existing_text']['path']:x['existing_text']['sha256']for x in rows}},
                rows=rows,counts=dict(original_denominator=44,pdf_events_checked=len(rows),flattened_pdf_basic_denominator_candidates=count,
                all_formats_basic_denominator_candidates=basic_count,original_unambiguous_explicit_annual_pair_locators=inventory['counts']['by_status'].get('explicit_annual_pair_locator_present',0)),
                raw_financial_values_recorded=False,activity={'features':0,'ranks':0,'ratios':0,'returns':0,'trials':0},
                html_qualification='The 12 NVIDIA adjusted-context flags concern the broad preceding context and require local statement-boundary review; they do not establish that the candidate share rows are non-GAAP. The seven Microsoft missing-unit flags are locator limitations, not proof that no units are disclosed.')
    base.write_once(OUT,result);print(json.dumps({'output':OUT,'sha256':base.sha(base.ROOT/OUT),'counts':result['counts']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.test:
        r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        if not r.wasSuccessful():raise SystemExit(1)
    elif a.write:build()
    else:p.error('Choose --test or --write')
