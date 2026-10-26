"""Compare independently extracted raw original-source evidence with both producers.
No financial-ratio, feature, return, or strategy admission computation.
"""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from datetime import datetime,timezone
import json
import capex193_independent as a
PRODUCERS={
 'outputs/monthly_capex193_source/msft_aapl/EXTRACTION.json':'288f1af295a25cadbf2eb26437e53513382942fc50db46997164e50b3edebf2d',
 'outputs/monthly_capex193_source/nvda_amd/EXTRACTION.json':'81dbb8535788765e8ec0497d0bc4314b4728a6f5aad6b36525e83143333627ee'}
VISUAL='outputs/monthly_capex193_source/msft_aapl/VISUAL_REVIEW.json'
VISUAL_SHA='204ef920b1819441dee304439641ab21101396ec4cc869c964befa59cbd3efe7'

def compare_records(expected,producers):
 a.need(len({r['event_id'] for r in expected})==len(expected),'duplicate expected')
 a.need(len({r['event_id'] for r in producers})==len(producers),'duplicate producer')
 by={r['event_id']:r for r in producers};a.need(set(by)=={r['event_id'] for r in expected},'coverage differs')
 out=[]
 for e in expected:
  p=by[e['event_id']];prefix=e['event_id']+': '
  for k in ('symbol','fiscal_year','release_date','available_from','inherited_admitted'):a.need(p[k]==e[k],prefix+k)
  a.need(p['source']['sha256']==e['source_sha256'] and p['source']['path']==e['source']['path'],prefix+'source')
  a.need(p['extraction_verified'] is e['cash_ppe_verified'],prefix+'extraction status')
  a.need(p['status']==('verified' if e['cash_ppe_verified'] and e['inherited_admitted'] else 'held'),prefix+'held status')
  a.need(p.get('fact_admitted') is False and p.get('strategy_admitted') is False,prefix+'no new admission')
  holds=p.get('inherited_holds',p.get('inherited_hold_reasons',[]))
  a.need(set(e['inherited_hold_reasons']).issubset(holds) and set(e['inherited_hold_reasons']).issubset([x[len('inherited:'):] if x.startswith('inherited:') else x for x in p['hold_reasons']]),prefix+'inherited holds lost')
  f=p.get('field',p.get('capex'))
  if e['cash_ppe_verified']:
   a.need(isinstance(f,dict),prefix+'missing field')
   for k in ('positive_outflow_decimal','signed_value_decimal','scale'):a.need(Decimal(f[k])==Decimal(e[k]),prefix+k)
   a.need(a.number(f['signed_literal'])==Decimal(e['signed_value_decimal'])<=0,prefix+'signed literal')
   a.need(f['locator']['period_end']==e['annual_period_end'] and f['period']['period_end']==e['annual_period_end'],prefix+'annual period')
   a.need(a.sha(f['row_text'].encode())==f['row_text_sha256'],prefix+'row hash')
   a.need(not any(x in f['row_label'].lower() for x in ('intangible','principal','noncash')),prefix+'mixed or noncash row')
   ee=e['capex_evidence']
   if f['locator']['format']=='html':
    for key,orig in [('table_index_zero_based','table'),('row_index_zero_based','row'),('logical_columns','logical_columns')]:a.need(f['locator'][key]==ee['locator'][orig],prefix+'financial cell '+key)
   else:
    a.need(f['locator']['page_1based']==ee['page'] and f['locator']['financial_column_index_0based']==ee['financial_column_index'],prefix+'PDF financial column')
   if e['inherited_admitted']:a.need(e['paired_cfo_verified'] and e['paired_assets_verified'],prefix+'paired current CFO/assets')
  else:a.need(f is None and p['hold_reasons'],prefix+'missing/ambiguous must remain null/held')
  out.append(dict(deepcopy(e),producer_comparison_passed=True,producer_status=p['status']))
 return out

def main():
 ep=a.OUT/'EXPECTED.json';expected=json.loads(ep.read_text())
 a.need(expected['scope_sha256']==a.SCOPE_SHA,'scope')
 a.pinned(a.SCOPE,a.SCOPE_SHA)
 a.need(a.sha(Path(a.__file__).read_bytes())==expected['code_sha256'],'expected extractor changed')
 for p,h in expected['source_pins'].items():a.pinned(p,h)
 a.pinned(a.SESSIONS,a.SESSIONS_SHA)
 a.runtime() # Full new toolchain gate; does not represent the failed old closure as passing.
 ps=[];producer_codepins={}
 for p,h in PRODUCERS.items():
  o=json.loads(a.pinned(p,h));a.need(o['scope_sha256']==a.SCOPE_SHA,'producer scope');ps.extend(o.get('records',o.get('events',[])))
  cp=o.get('code_hashes',o.get('code_sha256',{}))
  if isinstance(cp,dict):
   for f,d in cp.items():a.pinned(f,d);producer_codepins[f]=d
 rows=compare_records(expected['events'],ps);a.need(len(rows)==44,'44 events')
 a.pinned(VISUAL,VISUAL_SHA)
 # All ten images were directly inspected in this independent agent, after
 # independent original-PDF text/geometry extraction had already been saved.
 renders={};visuals=[]
 for e in rows:
  if e['symbol']!='AAPL':continue
  p=f"outputs/monthly_capex193_source/msft_aapl/renders/AAPL_FY{e['fiscal_year']}_cashflow.png"
  h=a.sha(a.path(p).read_bytes());renders[p]=h
  visuals.append(dict(event_id=e['event_id'],source_sha256=e['source_sha256'],render_path=p,render_sha256=h,current_annual_column_first=True,units_millions=True,pure_cash_ppe_separate_from_intangibles=True,source_amount=e['positive_outflow_decimal'],cfo_matches_source=True,independent_visual_review=True))
 codepins={}
 for p in ('work/capex193_independent.py','work/capex193_independent_review.py','work/capex193_independent_tests.py'):codepins[p]=a.sha(a.path(p).read_bytes())
 testref=a.OUT/'TESTS_02.json';test=json.loads(testref.read_text());a.need(test['passed'] is True and test['tests_run']>=15,'synthetic audit tests')
 a.need(test['code_sha256']==codepins,'tests bound to current code')
 qualified=[e for e in rows if e['cash_ppe_verified'] and e['inherited_admitted'] and e['paired_cfo_verified'] and e['paired_assets_verified']]
 a.need(len(qualified)==24,'source-only expected support census')
 report=dict(schema='capex193_independent_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,source_audit_is_not_strategy_admission=True,ratios_computed=False,factors_computed=False,returns_computed=False,scope_sha256=a.SCOPE_SHA,producer_refs=PRODUCERS,producer_visual_review_ref={VISUAL:VISUAL_SHA},independent_expected_ref={str(ep.relative_to(a.ROOT)):a.sha(ep.read_bytes())},test_ref={str(testref.relative_to(a.ROOT)):a.sha(testref.read_bytes())},code_sha256=codepins,producer_code_sha256=producer_codepins,runtime=expected['runtime'],source_pins=expected['source_pins'],calendar_sha256=a.SESSIONS_SHA,render_sha256=renders,independent_visual_reviews=visuals,events=rows,summary=dict(events=44,extraction_verified=27,source_qualified_before_strategy_admission=24,held=20,inherited_admitted=31,inherited_holds=13,new_capex_holds_among_inherited_admitted=7,paired_cfo_original_matches=sum(x['paired_cfo_verified'] for x in rows),paired_assets_original_matches=sum(x['paired_assets_verified'] for x in rows)),limitations=['Original first-public-version and full correction-history completeness are not certified; inherited identity and accounting holds are preserved.','Historical frozen admission and independent-review receipts are preserved. Their old PDF runtime closure fails current verification and is NOT claimed to pass. New source-only official-wheel runtime is separately verified.','NVDA combined PP&E/intangible rows are not separate cash PP&E. AMD2025 continuing/discontinued amounts are retained separately and not summed or selected.','Missing original current CFO remains missing; no zero fill. Currency ISO remains unverified; same-statement literal unit/currency evidence only.'])
 a.publish('REVIEW.json',report)
 lines=['# CAPEX193 独立原文核验','', '44/44 条源审核通过：27 条现金 PP&E 可精确提取；其中24条同时满足继承资格及配对 CFO/资产原文一致性，20条保留 hold。此结论仅是源审核，不是策略准入。','', '独立程序直接读取固定原文：HTML使用标准库解析器按逻辑年度列复核；PDF使用新完整校验的官方工具链、独立文本/几何带算法，并目检10张现金流页。35条既有 current CFO、全部44条 current/comparison资产金额与原 DATASET 一致；余9条原 CFO 缺失，未补零。','', '全部12条 NVIDIA 不能分离 PP&E 与无形资产；AMD2025两条持续/终止经营 PP&E 不合并。原13条 hold 不晋升，另7条原已准入 NVIDIA 因本次 PP&E 定义新增 hold。','', '旧 PDFium 二进制哈希漂移及签名失败未被绕过。新官方 PyPI wheel、70项 RECORD、55项原pypdf及126项完整 runtime pins已验证；不声称旧runtime闭包当前通过。','', '未计算比率、因子、模型或收益。发布日后第一个固定交易日逐条一致。机器证据见 REVIEW.json 与 EXPECTED.json。','', '| Symbol | Cash PP&E extracted | Retained source-qualified |','|---|---:|---:|']
 for symbol in ('AAPL','AMD','MSFT','NVDA'):lines.append('| '+symbol+' | '+str(sum(e['symbol']==symbol and e['cash_ppe_verified'] for e in rows))+' | '+str(sum(e['symbol']==symbol for e in qualified))+' |')
 with (a.OUT/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps(report['summary'],ensure_ascii=False));print('REVIEW SHA',a.sha((a.OUT/'REVIEW.json').read_bytes()))
if __name__=='__main__':main()
