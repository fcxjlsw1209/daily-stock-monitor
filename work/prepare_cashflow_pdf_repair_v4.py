"""Prepare an isolated, bounded-kerning source repair from exact frozen V3.

No financial extraction. Only code, synthetic fixtures and failure provenance
are written; the test/draft/freeze/build phases remain separate commands.
"""
from pathlib import Path
import hashlib
import json
import cashflow_source_preflight_v1 as base

ROOT=Path(__file__).resolve().parents[1]
PINS={
 'work/cashflow_pdf_repair_v3.py':'8b8eb5c0d47ccbf43a64394ec565af61e78286f6db7e14e5a153d568facac3c8',
 'work/test_cashflow_pdf_repair_v3.py':'b490df057d033ed3b13facd03817fc3f4dad76117e127e09581695d47c1d6803',
 'work/prepare_cashflow_pdf_repair_v3.py':'c91a290de4fdcd4aabdf895f02cdc3b243a78986a5abe71484b557f9d2ded2db',
 'work/diagnose_cashflow_pdf_v3_failure.py':'ef38715cedadd88a453ab3ef54d3d45441e7a16ba099840586a3372ed10d1918',
 'work/cashflow_pdf_repair_v3_review.json':'1bd40f9644f025feecf7c8d47b2ca69da27fc2705fdc588b9189a96be75870ba',
 'outputs/monthly_cashflow_pdf_repair_v3/PLAN.json':'a9c63e4ba60e448843d7462bfd3533cd3f729b87383b30efc7a8553421021f1f',
 'outputs/monthly_cashflow_pdf_repair_v3/PLAN_DRAFT.json':'54040883b028cae4c3bacaf6d9428dc2c9d92ddbcb80f3c7f48a3aee79c396ac',
 'outputs/monthly_cashflow_pdf_repair_v3/TESTS.json':'5fdc91f5d0ee32c5998d252bd62a29e39479b84f824ea22033a46e771099035c'}
base.safe.verify_hashes(ROOT,PINS)
for name in ('RAW_FACTS.json','AUDIT.json','MANIFEST.json'):
 base.require(not (ROOT/'outputs/monthly_cashflow_pdf_repair_v3'/name).exists(),'V3 failure record expects no published accepted outputs')

FAILURE_PATH='outputs/monthly_cashflow_pdf_repair_v4/PREDECESSOR_FAILURE.json'
failure=dict(schema_version=1,stage='failed_predecessor_preserved_not_a_source_admission',
 predecessor='monthly_cashflow_pdf_repair_v3',predecessor_hashes=PINS,
 build_exit_code_reported_by_root=1,build_error_reported_by_root='V3 cannot lose a recovered V2 value',
 accepted_v3_outputs_observed_absent=['RAW_FACTS.json','AUDIT.json','MANIFEST.json'],
 cause='V3 rejects every negative horizontal glyph gap. Root diagnosis found consecutive PDFium source indices for a 7 followed by comma with approximately -0.1359 point kerning overlap.',
 affected_scope_reported_by_root=['AAPL FY2020 net income','AAPL FY2021 comparison net income'],
 evidence_scope='Failure and affected rows are recorded from the parent task diagnosis. This preparation does not re-extract real financial cells.',
 proposed_rule='Retain consecutive source indices and upper gap <=0.75 times the greater adjacent glyph height; allow negative gap only down to -min(0.5 point,0.25 times the smaller adjacent glyph width).',
 no_value_based_selection=True,prior_fifty_value_guard_retained=True,actual_financial_cells_extracted=0,
 fact_admitted=False,model_admitted=False,strategy_admitted=False)
failure_hash=hashlib.sha256(base.safe.encoded(failure)).hexdigest()

text=(ROOT/'work/cashflow_pdf_repair_v3.py').read_text()
text=text.replace('cashflow_pdf_repair_v3','cashflow_pdf_repair_v4')
text=text.replace('revision3 for font-dependent numeral spacing','revision4 for bounded glyph kerning')
text=text.replace('V3 cannot lose','V4 cannot lose').replace('V3 changed recovered','V4 changed recovered')
needle='require = v1.require'
base.require(text.count(needle)==1,'Unique pin insertion anchor')
text=text.replace(needle,'V3_PINS = '+repr(PINS)+'\nREPAIR_PINS = '+repr({FAILURE_PATH:failure_hash})+'\nPREP = "work/prepare_cashflow_pdf_repair_v4.py"\n'+needle)
text=text.replace('for p in (SELF,TEST)}','for p in (SELF,TEST,PREP)}')
text=text.replace(' v1.safe.verify_hashes(root,V2_PINS)\n',' v1.safe.verify_hashes(root,V2_PINS)\n v1.safe.verify_hashes(root,V3_PINS)\n v1.safe.verify_hashes(root,REPAIR_PINS)\n',1)
text=text.replace('source_hashes.update(V2_PINS)','source_hashes.update(V2_PINS);source_hashes.update(V3_PINS);source_hashes.update(REPAIR_PINS)')
text=text.replace('NUMERIC_GAP_FONT_HEIGHT_RATIO = 0.75','NUMERIC_GAP_FONT_HEIGHT_RATIO = 0.75\nNUMERIC_OVERLAP_MIN_WIDTH_RATIO = 0.25\nNUMERIC_OVERLAP_CAP_PT = 0.5')
text=text.replace('well as a bounded geometric gap. Whitespace, separate text runs and\n separated columns cannot be joined merely to match pypdf values.',
 'well as a bounded geometric gap. Small negative gaps from font kerning\n are bounded by both the smaller glyph width and an absolute point cap.\n Original whitespace and nonconsecutive indices are never bridged;\n independent row text and date-column geometry remain required.')
needle="   gap=c['bbox'][0]-prev['bbox'][2]\n   join=(c['index']==prev['index']+1 and\n         0 <= gap <= NUMERIC_GAP_FONT_HEIGHT_RATIO*height)"
replacement="""   gap=c['bbox'][0]-prev['bbox'][2]
   width=min(prev['bbox'][2]-prev['bbox'][0],c['bbox'][2]-c['bbox'][0])
   overlap=min(NUMERIC_OVERLAP_CAP_PT,NUMERIC_OVERLAP_MIN_WIDTH_RATIO*width)
   join=(c['index']==prev['index']+1 and
         -overlap <= gap <= NUMERIC_GAP_FONT_HEIGHT_RATIO*height)"""
base.require(text.count(needle)==1,'Unique numeric grouping replacement')
text=text.replace(needle,replacement)
text=text.replace('numeric_gap_font_height_ratio=NUMERIC_GAP_FONT_HEIGHT_RATIO,',
 'numeric_gap_font_height_ratio=NUMERIC_GAP_FONT_HEIGHT_RATIO,\n  numeric_overlap_min_width_ratio=NUMERIC_OVERLAP_MIN_WIDTH_RATIO,numeric_overlap_cap_pt=NUMERIC_OVERLAP_CAP_PT,')
text=text.replace('Consecutive original PDFium character indices and nonnegative gap <=0.75 times adjacent maximum glyph height; original whitespace and column geometry preserved.',
 'Consecutive original PDFium character indices; gap between -min(0.5pt,0.25 times adjacent minimum glyph width) and 0.75 times adjacent maximum glyph height; original whitespace and column geometry preserved.')
text=text.replace('v2_frozen_artifacts_unchanged=True,','v2_frozen_artifacts_unchanged=True,v3_failed_frozen_artifacts_unchanged=True,')

test=(ROOT/'work/test_cashflow_pdf_repair_v3.py').read_text().replace('cashflow_pdf_repair_v3','cashflow_pdf_repair_v4')
extra='''
class BoundedKerningTests(unittest.TestCase):
 def pair(self,gap=-0.1359,width_left=4.0,width_right=1.5,delta=1):
  return [dict(text='7',index=20,bbox=[0,0,width_left,8]),
          dict(text=',',index=20+delta,bbox=[width_left+gap,0,width_left+gap+width_right,8])]
 def test_small_observed_kerning_shape_passes(self):
  self.assertEqual(len(m.numeric_groups(self.pair())),1)
 def test_small_kerning_in_complete_synthetic_row_passes(self):
  p=page(('27,345','222'))
  for i,(ch,b) in enumerate(p.chars):
   if ch=='7' and b[1]==400 and b[0]==385:
    p.chars[i]=(ch,(b[0],b[1],390.1359,b[3]))
  self.assertEqual(geometry(p,('27,345','222'))['selected_cell']['text'],'27,345')
 def test_overlap_exceeding_smaller_width_limit_splits(self):
  self.assertEqual(len(m.numeric_groups(self.pair(gap=-0.11,width_right=0.4))),2)
 def test_overlap_exceeding_absolute_point_limit_splits(self):
  self.assertEqual(len(m.numeric_groups(self.pair(gap=-0.6,width_left=8,width_right=8))),2)
 def test_half_point_overlap_at_bound_passes(self):
  self.assertEqual(len(m.numeric_groups(self.pair(gap=-0.5,width_left=4,width_right=4))),1)
 def test_full_glyph_superposition_rejected(self):
  self.assertEqual(len(m.numeric_groups(self.pair(gap=-4,width_left=4,width_right=4))),2)
 def test_small_overlap_never_bridges_source_index_gap(self):
  self.assertEqual(len(m.numeric_groups(self.pair(delta=2))),2)
 def test_small_overlap_never_reverses_source_order(self):
  self.assertEqual(len(m.numeric_groups(self.pair(delta=-1))),2)
 def test_huge_overlap_complete_row_rejected(self):
  p=page(('27,345','222'))
  for i,(ch,b) in enumerate(p.chars):
   if ch=='7' and b[1]==400 and b[0]==385:
    p.chars[i]=(ch,(b[0],b[1],392,b[3]))
  with self.assertRaises(ValueError):geometry(p,('27,345','222'))
 def test_unknown_glyph_not_normalized_by_kerning(self):
  chars=self.pair();chars[1]['text']='@'
  with self.assertRaises(ValueError):m.numeric_tail(chars)
 def test_positive_gap_limit_is_unchanged(self):
  self.assertEqual(len(m.numeric_groups(self.pair(gap=6.1))),2)
 def test_negative_amount_keeps_sign_with_kerning(self):
  p=page(('-27,345','222'))
  for i,(ch,b) in enumerate(p.chars):
   if ch=='7' and b[1]==400 and b[0]==390:
    p.chars[i]=(ch,(b[0],b[1],395.1359,b[3]))
  self.assertEqual(geometry(p,('-27,345','222'))['selected_cell']['text'],'-27,345')

'''
anchor="\nif __name__=='__main__':"
base.require(test.count(anchor)==1,'Unique synthetic test insertion')
test=test.replace(anchor,'\n'+extra+anchor)

outputs={'work/cashflow_pdf_repair_v4.py':text,'work/test_cashflow_pdf_repair_v4.py':test}
for path in outputs:base.require(not (ROOT/path).exists(),'V4 output already exists: '+path)
base.require(not (ROOT/FAILURE_PATH).exists(),'V4 failure record already exists')
base.publish(ROOT,FAILURE_PATH,failure)
for path,content in outputs.items():
 with (ROOT/path).open('x') as stream:stream.write(content)
base.safe.verify_hashes(ROOT,PINS)
print(json.dumps(dict(created=list(outputs)+[FAILURE_PATH],original_frozen_v3_unchanged=True,actual_source_cells_extracted=0)))
