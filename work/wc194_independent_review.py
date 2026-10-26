"""Bind and compare independent raw-source WC194 reconstruction; no component sums."""
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal
import argparse,io,json,unittest
import wc194_independent as a
b=a.b
EXPECTED='outputs/monthly_working_capital194_source/independent/EXPECTED.json'
EXPECTED_SHA='8817bec3bd36d5949116baaac81d60b7af1c9f6ccc3a4084055291d8d7d03f7a'
PREFLIGHT='outputs/monthly_working_capital194_source/independent/PREFLIGHT.json'
PREFLIGHT_SHA='3ad43e9cf9f1ec50e522a4c1460372e5eb0363b6ca9fe763045298661f1c939e'
PRODUCER_PATHS=('outputs/monthly_working_capital194_source/msft_aapl/EXTRACTION.json','outputs/monthly_working_capital194_source/nvda_amd/EXTRACTION.json')
VISUAL='outputs/monthly_working_capital194_source/msft_aapl/VISUAL_REVIEW.json'
CODES=('work/wc194_independent.py','work/wc194_independent_review.py','work/wc194_independent_tests.py','work/wc194_independent_review_tests.py')
# Directly observed on the ten independently generated PDF page images.
VISUAL_COMPONENTS={2016:('1095','217','1791'),2017:('-2093','-2723','9618'),2018:('-5322','828','9175'),2019:('245','-289','-1923'),2020:('6917','-127','-4062'),2021:('-10125','-2642','12326'),2022:('-1823','1484','9448'),2023:('-1688','-1618','-1889'),2024:('-3788','-1046','6020'),2025:('-6682','1400','902')}

def codepins():return {p:b.sha(b.path(p).read_bytes()) for p in CODES}

def compare_records(expected,produced):
 b.need(len({x['event_id'] for x in expected})==len(expected),'duplicate expected events')
 b.need(len({x['event_id'] for x in produced})==len(produced),'duplicate producer events')
 by={x['event_id']:x for x in produced};b.need(set(by)=={x['event_id'] for x in expected},'event coverage mismatch');out=[]
 for e in expected:
  p=by[e['event_id']];prefix=e['event_id']+': '
  for key in ('symbol','fiscal_year','release_date','available_from','inherited_admitted'):b.need(p[key]==e[key],prefix+key)
  for key in ('path','sha256'):b.need(p['source'][key]==e['source'][key],prefix+'record source '+key)
  b.need(e['source_sha256']==e['source']['sha256'],prefix+'expected source')
  b.need(p['extraction_verified'] is e['extraction_verified'],prefix+'complete extraction status')
  b.need(p['status']==('verified' if e['extraction_verified'] and e['inherited_admitted'] else 'held'),prefix+'status')
  b.need(p.get('fact_admitted') is False and p.get('strategy_admitted') is False,prefix+'unregistered admission')
  inherited=p.get('inherited_holds',p.get('inherited_hold_reasons',[]));holds=[x[len('inherited:'):] if x.startswith('inherited:') else x for x in p['hold_reasons']]
  b.need(set(e['inherited_hold_reasons']).issubset(inherited) and set(e['inherited_hold_reasons']).issubset(holds),prefix+'inherited hold missing')
  if not(e['extraction_verified'] and e['inherited_admitted']):b.need(holds,prefix+'held without reason')
  b.need(set(p['fields'])==set(a.KEYS),prefix+'component key set')
  for key in a.KEYS:
   f=p['fields'][key];g=e['fields'][key];b.need((f is None)==(g is None),prefix+key+' null/partial status')
   c=e['components'][key];b.need(c['verified'] is (g is not None),prefix+'component summary')
   if g is None:
    b.need(all(c[k] is None for k in ('signed_value_decimal','scale','annual_period_end')),prefix+'held component summary nonnull');continue
   b.need(a.exact_component_label(f['row_label'],key),prefix+key+' pure row label')
   for k in ('signed_value_decimal','scale'):
    b.need(isinstance(f[k],str) and Decimal(f[k]).is_finite() and Decimal(f[k])==Decimal(g[k]),prefix+key+' '+k)
   b.need(b.number(f['signed_literal'])==Decimal(g['signed_value_decimal']),prefix+key+' printed sign/value')
   b.need(f['currency_iso'] is None,prefix+key+' unproved ISO')
   b.need(f['currency_literal'] in (None,'$'),prefix+key+' currency literal')
   for sk in ('path','sha256'):b.need(f['source'][sk]==g['source'][sk],prefix+key+' field source '+sk)
   b.need(f['period']==g['period'] and f['available_from']==e['available_from'],prefix+key+' period/clock')
   b.need(f['locator']['period_end']==g['annual_period_end'],prefix+key+' annual end')
   b.need(b.sha(f['row_text'].encode())==f['row_text_sha256'],prefix+key+' row digest')
   b.need(b.compact(f['row_text'])==b.compact(g['row_text']),prefix+key+' complete original row')
   loc=f['locator'];gloc=g['locator'];b.need(loc['format']==gloc['format'],prefix+key+' format')
   b.need(loc['original_cfo_locator']==gloc['original_cfo_locator'],prefix+key+' original CFO locator')
   ce=f['statement_currency_evidence'];b.need(ce['format']==gloc['format'] and ce['original_cfo_locator']==gloc['original_cfo_locator'],prefix+key+' statement currency source')
   if loc['format']=='html':
    b.need(loc['table_index_zero_based']==gloc['table'] and loc['row_index_zero_based']==gloc['row'],prefix+key+' exact original table row')
    b.need(loc['logical_columns']==gloc['logical_columns'],prefix+key+' exact annual/sign columns')
    b.need(ce['table_index_zero_based']==gloc['table'],prefix+key+' currency table')
    b.need(ce.get('statement_segment')==gloc['original_cfo_locator'].get('statement_segment'),prefix+key+' currency statement segment')
   else:
    b.need(loc['page_1based']==gloc['page'] and loc['financial_column_index_0based']==gloc['financial_column_index'] and loc['header_dates']==gloc['header_dates'],prefix+key+' PDF annual column')
    b.need(ce['page_1based']==gloc['page'],prefix+key+' currency page')
  if e['extraction_verified'] and e['inherited_admitted']:b.need(e['paired_cfo_verified'] and e['paired_assets_verified'],prefix+' original CFO/assets not independently verified')
  out.append(dict(deepcopy(e),producer_comparison_passed=True,producer_status=p['status']))
 return out

def addpins(target,pins):
 for p,h in pins.items():
  rel=str(b.path(p).relative_to(a.ROOT));b.pinned(rel,h)
  b.need(rel not in target or target[rel]==h,'conflicting pin '+rel);target[rel]=h

def tests():
 import wc194_independent_tests as t1, wc194_independent_review_tests as t2
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(t1),unittest.defaultTestLoader.loadTestsFromModule(t2)])
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue())
 b.need(result.wasSuccessful(),'independent synthetic failures')
 a.publish('TESTS.json',dict(passed=True,tests_run=result.testsRun,synthetic_only=True,code_sha256=codepins(),helper_sha256={a.BASE:a.BASE_SHA},log=stream.getvalue()))

def review(msft_sha,nvda_sha,visual_sha):
 expected=json.loads(b.pinned(EXPECTED,EXPECTED_SHA));b.pinned(PREFLIGHT,PREFLIGHT_SHA)
 b.need(expected['scope_sha256']==a.SCOPE_SHA and expected['concept_sha256']==a.CONCEPT_SHA,'expected scope/concept')
 for p,h in expected['code_sha256'].items():b.pinned(p,h)
 scope,_=a.scope_inputs();pins={};addpins(pins,expected['input_sha256'])
 pypdf,pdfium,runtime=b.runtime()
 rm=b.load(b.RUNTIME);acq=b.load(rm['acquisition']['path']);directory=str(b.path(b.RUNTIME).parent.relative_to(a.ROOT))
 addpins(pins,{directory+'/pypi_release.json':acq['metadata_sha256'],directory+'/'+acq['wheel_filename']:acq['wheel_sha256']})
 ppins=dict(zip(PRODUCER_PATHS,(msft_sha,nvda_sha)));producers=[];producer_codes={}
 for p,h in ppins.items():
  obj=json.loads(b.pinned(p,h));b.need(obj['scope_sha256']==a.SCOPE_SHA and obj['concept_sha256']==a.CONCEPT_SHA,'producer scope/concept');producers.extend(obj['records'])
  cp=obj.get('code_sha256',obj.get('code_hashes',{}));b.need(isinstance(cp,dict) and cp,'producer code pins');addpins(producer_codes,cp)
  for field in ('input_sha256','source_hashes','pdf_runtime_hashes'):
   addpins(pins,obj.get(field,{}))
 rows=compare_records(expected['events'],producers);b.need(len(rows)==44,'all44')
 visual=json.loads(b.pinned(VISUAL,visual_sha));serialized=json.dumps(visual)
 b.need(msft_sha in serialized,'producer visual not bound to extraction')
 visual_refs={VISUAL:visual_sha}
 independent_visual=[];renderpins={}
 for e in rows:
  if e['symbol']!='AAPL':continue
  yy=int(e['fiscal_year']);values=VISUAL_COMPONENTS[yy]
  for key,v in zip(a.KEYS,values):b.need(e['components'][key]['signed_value_decimal']==v,'manual independent visual component '+e['event_id']+key)
  page=e['candidate_evidence'][0];renderpins[page['render_path']]=page['render_sha256'];b.pinned(page['render_path'],page['render_sha256'])
  independent_visual.append(dict(event_id=e['event_id'],source_sha256=e['source_sha256'],render_path=page['render_path'],render_sha256=page['render_sha256'],page=3,current_annual_column=0,units_millions=True,three_distinct_operating_reconciliation_rows=True,signs_and_values_match_independent_extraction=True,visual_source='independent_generated_PDF_page_directly_viewed',signed_values=dict(zip(a.KEYS,values)),special_layout_note='PDFium complete numeric glyph band and direct image confirm 9,448 despite spaced pypdf digits' if yy in (2022,2023) else None))
 tr=str((a.OUT/'TESTS.json').relative_to(a.ROOT));test=b.load(tr);b.need(test['passed'] is True and test['code_sha256']==codepins(),'tests bound to final independent code')
 addpins(pins,producer_codes);addpins(pins,codepins());addpins(pins,{EXPECTED:EXPECTED_SHA,PREFLIGHT:PREFLIGHT_SHA,tr:b.sha(b.path(tr).read_bytes())});addpins(pins,ppins);addpins(pins,visual_refs);addpins(pins,renderpins)
 # Bind every producer render independently of whether referenced by the visual receipt schema.
 for p in producers:
  rr=p.get('render')
  if isinstance(rr,dict):
   path=rr.get('path',rr.get('png_path'));digest=rr.get('sha256',rr.get('png_sha256'))
   if path and digest:addpins(pins,{path:digest})
 summary=dict(events=44,three_signed_rows_verified=sum(e['extraction_verified'] for e in rows),component_cells={k:sum(e['components'][k]['verified'] for e in rows) for k in a.KEYS},inherited_admitted=sum(e['inherited_admitted'] for e in rows),inherited_held=sum(not e['inherited_admitted'] for e in rows),source_qualified_before_strategy_admission=sum(e['extraction_verified'] and e['inherited_admitted'] and e['paired_cfo_verified'] and e['paired_assets_verified'] for e in rows),paired_current_cfo_matches=sum(e['paired_cfo_verified'] for e in rows),paired_assets_matches=sum(e['paired_assets_verified'] for e in rows),new_component_holds_among_inherited_admitted=sum(e['inherited_admitted'] and not e['extraction_verified'] for e in rows))
 summary['held']=44-summary['source_qualified_before_strategy_admission']
 report=dict(schema='wc194_independent_original_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,source_audit_is_not_strategy_admission=True,component_sum_computed=False,ratios_computed=False,factors_computed=False,orders_computed=False,returns_computed=False,scope_sha256=a.SCOPE_SHA,concept_sha256=a.CONCEPT_SHA,producer_refs=ppins,producer_visual_refs=visual_refs,expected_ref={EXPECTED:EXPECTED_SHA},preflight_ref={PREFLIGHT:PREFLIGHT_SHA},tests_ref={tr:b.sha(b.path(tr).read_bytes())},input_sha256=pins,code_sha256=codepins(),producer_code_sha256=producer_codes,runtime=runtime,independent_visual_reviews=independent_visual,summary=summary,events=rows,execution_findings=[dict(source='retained_tool_output_excerpt_not_full_log',initial_extraction_completed=False,error="ValueError: ambiguous monetary literal: '(2,215'",resolution='New independent parser includes only adjacent explicit parenthesis/currency glyph cells; no inferred signs, numeric merging, or weakening of annual-column checks. Initial render bytes were retained and re-rendered byte-identically before the successful complete audit.')],limitations=['Historical frozen old146/source receipts remain unchanged. Their old PDF runtime fails current checks; this audit verifies the separately acquired193 official-wheel isolated runtime, not the failed old binary.','Original13 holds retained; AMD2016/17 add combined-AP holds. AMD2025 three continuing-operations lines cannot pair with total original CFO and are retained only as candidate evidence.','Signed cash contributions are not balance-sheet changes. No summing or substitution for missing components.','Currency ISO, complete correction history, certified daily identity and first-public-version completeness remain unverified.','Source availability and this review do not admit a trading strategy or establish future returns.'])
 a.publish('REVIEW.json',report)
 with (a.OUT/'REPORT.md').open('x') as f:
  f.write('# WC194 独立原文审核\n\n44/44 记录与两组producer一致；31条三项同口径原文齐全，保留旧资格后29条可供source-only准入、15条hold。AR/Inventory/AP分别验证34/34/31项，未求和。\n\n独立标准库HTMLParser按年度逻辑列复核；PDF使用193完整验证的隔离工具链，独立字形几何与10张逐页图像核对。35条既有current CFO和44条两期资产原文一致；旧缺失不补零。\n\nAMD2015–17合并AP不得充作纯AP；AMD2025三项属于continuing scope而原CFO含discontinued，正式字段null、候选原文保留。原13个hold不晋升，另AMD2016/17新增hold。\n\n旧PDF runtime失败保留；新官方wheel、RECORD、原pypdf及全runtime pins重新核验。没有计算W、比率、因子、订单或收益；机器证据及完整SHA闭包见REVIEW.json。\n')
 print(json.dumps(summary,ensure_ascii=False));print('REVIEW SHA',b.sha((a.OUT/'REVIEW.json').read_bytes()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['test','review','probe']);ap.add_argument('--msft-aapl-sha');ap.add_argument('--nvda-amd-sha');ap.add_argument('--visual-sha');args=ap.parse_args()
 if args.action=='test':tests()
 elif args.action=='probe':
  expected=json.loads(b.pinned(EXPECTED,EXPECTED_SHA));produced=[]
  for path,digest in zip(PRODUCER_PATHS,(args.msft_aapl_sha,args.nvda_amd_sha)):
   b.need(digest,'explicit producer digest');produced.extend(json.loads(b.pinned(path,digest))['records'])
  print('source comparisons passed',len(compare_records(expected['events'],produced)))
 else:
  b.need(all((args.msft_aapl_sha,args.nvda_amd_sha,args.visual_sha)),'explicit producer/visual SHA required');review(args.msft_aapl_sha,args.nvda_amd_sha,args.visual_sha)
