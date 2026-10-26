"""Isolated Apple PDF source repair, revision3 for font-dependent numeral spacing. Draft/freeze are metadata only.

Actual monetary parsing is allowed only by --build/--verify with an explicit
frozen plan hash. No arithmetic beyond exact decimal syntax normalization.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import re
import statistics
import sys
import cashflow_source_preflight_v1 as v1

ROOT = v1.ROOT
OUT = 'outputs/monthly_cashflow_pdf_repair_v3'
SELF = 'work/cashflow_pdf_repair_v3.py'
TEST = 'work/test_cashflow_pdf_repair_v3.py'
V1_PLAN = v1.OUT + '/PLAN.json'
V1_PINS = {
 V1_PLAN:'4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472',
 v1.OUT+'/RAW_FACTS.json':'4f6d6e694195a4986f911f7ba1defc24efc515b49aa510b5c4be61df78c45c87',
 v1.OUT+'/AUDIT.json':'43e908b15cf4ceb5fdfce700d74c6ff3788b906cff8f485c8c5e55026161c143',
 v1.OUT+'/MANIFEST.json':'ee636a289dab86fe4758b60f0073bbad5be2ecd327f2cce552235719ec10a068',
 v1.SELF:'c19b3b76bc2cabed6b4c1e3873be1a76e88e7675178debdc7b063e44afe5fbc9',
 v1.TEST:'584e0a55a83737770371807505a72993f9f32e67bde7850f18816713da432d02',
}
V2_PINS = {'outputs/monthly_cashflow_pdf_repair_v2/PLAN.json': '126e5e6afad0ac50013c04bda9813ca3d698b6d2ff6de4991243c4edf32272d0', 'outputs/monthly_cashflow_pdf_repair_v2/RAW_FACTS.json': '57eb60d0cada5f2ac521ec2c75db9992f52fbd2ec433fb70de1da2b79d68335c', 'outputs/monthly_cashflow_pdf_repair_v2/AUDIT.json': 'a5fa5b2ba3bbb854537e2c4a0ede819390e9d7594f334fbb5ec921e600bdc029', 'outputs/monthly_cashflow_pdf_repair_v2/MANIFEST.json': '825b41c1c4c6c42abad3f2cb996a1e5839c1737144eff6177c7c0aae1ca6e2c7', 'outputs/monthly_cashflow_pdf_repair_v2/TESTS.json': '037831723f80fb7fc3268ab1f0882c804a2abc75ad188db1a8d4333f1b3c7d35', 'work/cashflow_pdf_repair_v2.py': '4e7f5c0868c080847c8a290e17df4cc68d91d556aa26e28332737b86253869b5', 'work/test_cashflow_pdf_repair_v2.py': 'c275c193228b3725845fda85fab024aa1f12daebce3ba7e3c4927d7b6de887ba', 'work/cashflow_pdf_repair_v2_review.json': '9890f09e209d22b4bc88f3d594cd19e5eaffc1260b19ab1928640e731941a9cb'}
require = v1.require
compact = lambda s: re.sub(r'\s+', '', s).lower()
BASELINE_TOLERANCE_PT = 4.0
TOKEN_GAP_PT = 2.0


def code_hashes(root):
 return {p:v1.sha(root,p) for p in (SELF,TEST)}


def verified_inputs(root):
 v1.safe.verify_hashes(root,V1_PINS)
 v1.safe.verify_hashes(root,V2_PINS)
 plan=v1.read(root,V1_PLAN,V1_PINS[V1_PLAN])
 v1.safe.verify_hashes(root,plan['source_hashes'])
 v1.safe.verify_hashes(root,plan['source_code_hashes'])
 events=[e for e in plan['events'] if e['symbol']=='AAPL']
 require(len(events)==10 and len({e['event_id'] for e in events})==10,'Exact ten Apple events required')
 return plan,events


def monetary_unit(lines):
 """Only explicit primary monetary units; an exact share-count exception is separate."""
 found=[]
 for line in lines:
  value=compact(line).strip('()')
  if not value.startswith('inmillion') and not value.startswith('inthousand'):continue
  match=re.fullmatch(r'in(millions|thousands)(?:,exceptnumberofshares,?whicharereflectedinthousands,?andparvalue)?',value)
  require(match is not None,'unsupported_or_conflicting_monetary_unit')
  found.append((str({'millions':1000000,'thousands':1000}[match[1]]),line.strip()))
 require(len(found)==1,'unique_explicit_monetary_unit_required')
 return found[0]


def inventory(texts,periods):
 """Locate metadata only. Do not parse, retain, hash or emit monetary row tails."""
 candidates=v1.empty_candidates();gaps=[]
 for page,text in enumerate(texts):
  lines=text.splitlines();head=compact(' '.join(lines[:12]))
  cf='consolidatedstatementsofcashflows' in head
  bs='consolidatedbalancesheets' in head
  if cf==bs:continue
  try:
   require('appleinc.' in head,'missing_issuer_heading')
   require(not any(x in head for x in ('nongaap','non-gaap','proforma','asadjusted')),'nonformal_statement')
   scale,unit_literal=monetary_unit(lines[:8])
   dates,header_line,h1,h2=v1.pdf.headers(lines,2)
   require(len(set(dates))==2,'distinct_annual_dates_required')
   if cf:
    require('twelvemonthsended' in head or '12monthsended' in head or 'yearsended' in head,'missing_full_year_duration')
    require(not any(x in head for x in ('threemonthsended','sixmonthsended','ninemonthsended','3monthsended','6monthsended','9monthsended')),'mixed_or_interim_duration')
   require('$' in text or '/dollarsign' in text,'missing_literal_currency')
  except ValueError as ex:
   gaps.append(dict(page_1based=page+1,reason=str(ex)));continue
  matched={field:[] for field in v1.FIELDS}
  for i,line in enumerate(lines):
   for field,labels in v1.PDF_LABELS.items():
    for label in labels:
     if v1.pdf_row_match(line,label):matched[field].append((i,label))
  ni=matched['net_income'];cfo=matched['operating_cash_flow']
  adjustment=[i for i,line in enumerate(lines) if 'adjustmentstoreconcile' in compact(line)]
  ni_scope=len(ni)==len(cfo)==1 and any(ni[0][0]<i<cfo[0][0] for i in adjustment)
  for field in v1.FIELDS:
   if (field=='total_assets')!=bs:continue
   if field=='net_income' and not ni_scope:
    gaps.append(dict(page_1based=page+1,field=field,reason='unique_indirect_method_start_required'));continue
   for i,label in matched[field]:
    for side in v1.SIDES:
     wanted=periods[side]['period_end']
     if dates.count(wanted)!=1:continue
     candidates[side][field].append(dict(format='pdf',page_1based=page+1,line_1based=i+1,
      financial_column_index_0based=dates.index(wanted),row_label=label,header_dates=dates,
      header_literals=[h1,h2],date_header_line_1based=header_line,
      statement_title='consolidated statements of cash flows' if cf else 'consolidated balance sheets',
      unit_literal=unit_literal,scale=scale,currency_literal='$',period_end=wanted,
      ni_scope='indirect_method_consolidated_start' if field=='net_income' else None))
 return dict(candidates=candidates,metadata_gaps=gaps,accounting_review_flags=v1.accounting_flags(' '.join(texts)))


def bbox(chars):
 return [min(c['bbox'][0] for c in chars),min(c['bbox'][1] for c in chars),
         max(c['bbox'][2] for c in chars),max(c['bbox'][3] for c in chars)]


def center_y(c):return (c['bbox'][1]+c['bbox'][3])/2


def physical_rows(textpage):
 """Independent PDFium characters grouped by physical baseline, not phrase hits."""
 chars=[]
 for index in range(textpage.count_chars()):
  ch=textpage.get_text_range(index,1)
  if not ch or ch.isspace():continue
  require(len(ch)==1,'unmapped_PDFium_character')
  box=list(textpage.get_charbox(index))
  require(box[2]>box[0] and box[3]>box[1],'unlocated_PDFium_character')
  chars.append(dict(text=ch,index=index,bbox=box))
 rows=[]
 for c in sorted(chars,key=lambda c:(-center_y(c),c['bbox'][0])):
  eligible=[row for row in rows if abs(center_y(c)-statistics.median(center_y(x) for x in row))<BASELINE_TOLERANCE_PT]
  require(len(eligible)<=1,'ambiguous_PDFium_baseline')
  if eligible:eligible[0].append(c)
  else:rows.append([c])
 for row in rows:row.sort(key=lambda c:(c['bbox'][0],c['index']))
 return rows


def row_text(row):return ''.join(c['text'] for c in row)


def groups(chars):
 out=[]
 for c in chars:
  if not out or c['bbox'][0]-out[-1][-1]['bbox'][2]>TOKEN_GAP_PT:out.append([])
  out[-1].append(c)
 return out



NUMERIC_GAP_FONT_HEIGHT_RATIO = 0.75


def numeric_groups(chars):
 """Respect original character adjacency and font-scaled spacing.

 A narrow numeral can have more than 2pt of side-bearing before a comma
 or next digit. Joining requires consecutive original PDFium indices as
 well as a bounded geometric gap. Whitespace, separate text runs and
 separated columns cannot be joined merely to match pypdf values.
 """
 out=[]
 for c in chars:
  join=False
  if out:
   prev=out[-1][-1]
   height=max(prev['bbox'][3]-prev['bbox'][1],c['bbox'][3]-c['bbox'][1])
   gap=c['bbox'][0]-prev['bbox'][2]
   join=(c['index']==prev['index']+1 and
         0 <= gap <= NUMERIC_GAP_FONT_HEIGHT_RATIO*height)
  if not join:out.append([])
  out[-1].append(c)
 return out


def numeric_tail(chars):
 """Syntax only, with source glyphs preserved. '!' is never interpreted as USD.

 PDFium can map a separately positioned currency-font glyph to '!'. Ignore
 standalone currency-like decorations for numeral comparison, recording them;
 currency itself must independently be literal $ or /dollarsign in pypdf.
 """
 cells=[];decorations=[]
 for group in numeric_groups(chars):
  if group[0]['text'] in ('$', '!'):
   decorations.append(dict(text=group[0]['text'],bbox=group[0]['bbox'],index=group[0]['index']))
   group=group[1:]
  if not group:continue
  text=row_text(group)
  v1.pdf.decimal_cell(text)
  cells.append(dict(text=text,bbox=bbox(group),char_indices=[c['index'] for c in group]))
 return cells,decorations


def geometry_rows(rows,loc,expected_cells):
 """Whole financial-row identity plus independently located date-column geometry."""
 found=[]
 label=compact(loc['row_label'])
 for row in rows:
  full=row_text(row)
  if not full.lower().startswith(label):continue
  labelchars=row[:len(label)]
  try:
   cells,decorations=numeric_tail(row[len(label):])
  except ValueError:
   # A prose continuation cannot establish a competing financial row. A
   # numeric-looking malformed row must not disappear in favor of another.
   tail=full[len(label):]
   require(not tail or tail[0] not in '0123456789($!-','malformed_competing_financial_row')
   continue
  found.append((row,labelchars,cells,decorations))
 require(len(found)==1,'unique_exact_PDFium_financial_row_required')
 row,labelchars,cells,decorations=found[0]
 require(len(cells)==len(expected_cells)==len(loc['header_dates']),'PDFium_exact_column_count_required')
 require([c['text'] for c in cells]==expected_cells,'PDF_extractors_disagree_on_row')
 labelbox=bbox(labelchars)
 require(all(c['bbox'][0]>labelbox[2] for c in cells),'amount_must_be_right_of_label')
 headers=[]
 for literal in loc['header_literals']:
  matches=[r for r in rows if compact(row_text(r))==compact(literal)]
  require(len(matches)==1,'unique_exact_PDFium_date_header_required')
  headers.append(matches[0])
 date_row,year_row=headers
 yeargroups=groups(year_row)
 require([row_text(g) for g in yeargroups]==[d[:4] for d in loc['header_dates']],'PDFium_header_year_order_mismatch')
 years=[bbox(g) for g in yeargroups]
 dates=bbox(date_row);yb=bbox(year_row)
 require(dates[1]>yb[3] and yb[1]>bbox(row)[3],'headers_must_be_above_financial_row')
 centers=[(b[0]+b[2])/2 for b in years]
 require(centers==sorted(centers) and len(set(centers))==len(centers),'ambiguous_header_column_order')
 boundaries=[labelbox[2]]+[(a+b)/2 for a,b in zip(centers,centers[1:])]+[float('inf')]
 parts=v1.pdf.DATE.findall(loc['header_literals'][0]);dateboxes=[];offset=0
 require(len(parts)==len(years),'date_and_year_header_count_mismatch')
 for i,(month,day) in enumerate(parts):
  literal=month+day+',';piece=date_row[offset:offset+len(literal)];offset+=len(literal)
  require(row_text(piece)==literal,'date_header_character_mapping_mismatch')
  box=bbox(piece);cx=(box[0]+box[2])/2
  require(boundaries[i]<cx<boundaries[i+1],'date_and_year_columns_disagree')
  dateboxes.append(box)
 require(offset==len(date_row),'unmapped_date_header_characters')
 for i,c in enumerate(cells):
  require(c['bbox'][0]>boundaries[i] and c['bbox'][2]<boundaries[i+1],'amount_crosses_date_column_boundary')
 return dict(pdfium_label_bbox=labelbox,pdfium_all_row_cells=cells,pdfium_currency_like_decorations=decorations,
  pdfium_date_header_bbox=dates,pdfium_date_column_boxes=dateboxes,pdfium_year_header_boxes=years,
  selected_cell=deepcopy(cells[loc['financial_column_index_0based']]),
  baseline_tolerance_pt=BASELINE_TOLERANCE_PT,token_gap_pt=TOKEN_GAP_PT,
  numeric_gap_font_height_ratio=NUMERIC_GAP_FONT_HEIGHT_RATIO,
  numeric_requires_consecutive_source_character_indices=True)


def extract_cell(raw,texts,loc,pdf_plan):
 lines=texts[loc['page_1based']-1].splitlines();line=lines[loc['line_1based']-1]
 match=v1.pdf_row_match(line,loc['row_label']);require(match is not None,'frozen_exact_row_changed')
 cells=match[1].replace('/dollarsign','$').replace('$','').split()
 require(len(cells)==len(loc['header_dates']),'exact_pypdf_column_count_required')
 for value in cells:v1.pdf.decimal_cell(value)
 _,pdfium=v1.configure_pdf(pdf_plan);doc=pdfium.PdfDocument(raw);page=doc[loc['page_1based']-1];tp=page.get_textpage()
 try:geometry=geometry_rows(physical_rows(tp),loc,cells)
 finally:tp.close();page.close();doc.close()
 chosen=cells[loc['financial_column_index_0based']]
 return dict(displayed_amount=chosen,value_decimal=v1.pdf.decimal_cell(chosen),
  row_text=line,row_text_sha256=v1.safe.digest(line.encode()),pdfium_geometry=geometry,pdfium_text_agrees=True)


def schema():
 return dict(schema_version=2,stage='source_repair_only_not_admission',events=10,maximum_cells=60,
  fields=list(v1.FIELDS),sides=list(v1.SIDES),
  scope='Exactly the ten existing v1 Apple releases, all original rejected events and period metadata preserved; no new sources.',
  units='Primary explicit in millions/thousands, with only exact number-of-shares thousands/par-value exception; conflicting units fail.',
  metadata='Whitespace-elided title, annual duration and label matching; original header/unit literals retained. Draft/freeze never parse monetary tails.',
  numeric_tokenization='Consecutive original PDFium character indices and nonnegative gap <=0.75 times adjacent maximum glyph height; original whitespace and column geometry preserved.',
  row='Exact formal row label plus syntax-only financial tail; NI is unique indirect-method starting row before adjustment and CFO. Whole PDFium physical row must be unique, even if identical duplicates.',
  geometry=dict(baseline_tolerance_pt=BASELINE_TOLERANCE_PT,token_gap_pt=TOKEN_GAP_PT,
   policy='All PDFium nonblank characters on physical row are tested. Exact row cell count/text/order; both complete date and year headers independently found above row; entire numeral bbox must stay within its year-center midpoint column.',
   currency='Literal pypdf $ or /dollarsign required; isolated PDFium $/! glyph decorations recorded but do not certify ISO currency.'),
  gaps='Missing, ambiguous, malformed or disagreeing data stays null with explicit reason. No fallback, plausibility selection, manual values, imputation or sum.',
  admission='All cells/events remain fact_admitted/model_admitted/strategy_admitted false; source process verification only. Fresh accounting/identity/visual review still required.',
  arithmetic='Only exact decimal representation normalization, no ratios, signs, ranks, factors, fits, selections or returns.')


def draft(root):
 require(not v1.safe.local(root,OUT+'/PLAN.json').exists(),'Frozen repair is immutable')
 original,events=verified_inputs(root)
 tests=v1.read(root,OUT+'/TESTS.json');code=code_hashes(root)
 require(tests.get('passed') is True and tests.get('synthetic_only') is True and tests.get('source_code_hashes')==code,'Current passing synthetic tests required')
 require(all(tests.get(k)==0 for k in ('failures','errors','real_source_cells_extracted')),'Clean synthetic scope required')
 specs=[]
 for event in events:
  spec=deepcopy(event);raw=v1.safe.local(root,spec['source']['path']).read_bytes()
  require(v1.safe.digest(raw)==spec['source']['sha256'] and raw.startswith(b'%PDF-'),'Original PDF bytes required')
  spec['locator_inventory']=inventory(v1.pdf_texts(raw,original),spec['periods']);specs.append(spec)
 source_hashes=dict(original['source_hashes']);source_hashes.update(V1_PINS);source_hashes.update(V2_PINS)
 source_hashes[OUT+'/TESTS.json']=v1.sha(root,OUT+'/TESTS.json')
 plan=dict(schema_version=2,stage='draft_cashflow_pdf_repair_requires_review',created_utc=datetime.now(timezone.utc).isoformat(),
  runtime=sys.version,source_hashes=source_hashes,source_code_hashes=code,pdf_runtime=original['pdf_runtime'],
  pdf_runtime_hashes=original['pdf_runtime_hashes'],events=specs,expected_events=10,maximum_cells=60,policy=schema(),
  metadata_only_locators=True,financial_calculations_permitted=False,admission_permitted=False,network_requests_permitted=False)
 v1.publish(root,OUT+'/PLAN_DRAFT.json',plan)
 counts={s:{f:sum(len(e['locator_inventory']['candidates'][s][f])==1 for e in specs) for f in v1.FIELDS} for s in v1.SIDES}
 return dict(draft_sha256=v1.sha(root,OUT+'/PLAN_DRAFT.json'),unique_metadata_locators=counts,actual_source_cells_extracted=0)


def freeze(root,draft_sha):
 v1.expect_sha(draft_sha);plan=v1.read(root,OUT+'/PLAN_DRAFT.json',draft_sha)
 require(plan['stage']=='draft_cashflow_pdf_repair_requires_review','Reviewed draft required')
 verified_inputs(root);v1.safe.verify_hashes(root,plan['source_hashes'])
 require(plan['source_code_hashes']==code_hashes(root) and plan['runtime']==sys.version,'Draft code/runtime changed')
 plan['stage']='frozen_cashflow_pdf_repair_plan';plan['reviewed_draft_sha256']=draft_sha
 plan['source_hashes'][OUT+'/PLAN_DRAFT.json']=draft_sha
 v1.publish(root,OUT+'/PLAN.json',plan)
 return dict(plan_sha256=v1.sha(root,OUT+'/PLAN.json'),actual_source_cells_extracted=0)


def map_event(spec,raw,plan):
 texts=v1.pdf_texts(raw,plan);current=inventory(texts,spec['periods'])
 require(current==spec['locator_inventory'],'Frozen metadata locator inventory changed')
 result={k:deepcopy(v) for k,v in spec.items() if k!='locator_inventory'}
 result.update(fields={s:{} for s in v1.SIDES},gaps=[],metadata_gaps=deepcopy(current['metadata_gaps']),
  accounting_review_flags=deepcopy(current['accounting_review_flags']),fact_admitted=False,model_admitted=False,strategy_admitted=False,
  first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
 for side in v1.SIDES:
  for field in v1.FIELDS:
   locs=current['candidates'][side][field];result['fields'][side][field]=None
   reason='missing_formal_annual_cell' if not locs else 'context_or_column_ambiguous' if len(locs)!=1 else None
   if reason is None:
    loc=locs[0]
    try:
     value=extract_cell(raw,texts,loc,plan)
     result['fields'][side][field]=dict(value,scale=loc['scale'],currency_literal=loc['currency_literal'],currency_iso=None,
      period=deepcopy(spec['periods'][side]),available_from=spec['available_from'],source=deepcopy(spec['source']),locator=deepcopy(loc),fact_admitted=False)
    except (ValueError,IndexError,KeyError) as ex:reason='invalid_raw_cell: '+str(ex)
   if reason is not None:result['gaps'].append(dict(side=side,field=field,reason=reason,candidate_count=len(locs)))
 return result


def products(root,plan_sha):
 v1.expect_sha(plan_sha);plan=v1.read(root,OUT+'/PLAN.json',plan_sha);original,events=verified_inputs(root)
 require(plan['runtime']==sys.version and plan['stage']=='frozen_cashflow_pdf_repair_plan','Frozen plan/runtime mismatch')
 require(plan['source_code_hashes']==code_hashes(root),'Frozen code changed')
 require(all(plan[k] is False for k in ('financial_calculations_permitted','admission_permitted','network_requests_permitted')),'Source-only plan required')
 v1.safe.verify_hashes(root,plan['source_hashes']);v1.safe.verify_hashes(root,plan['source_code_hashes'])
 require(plan['expected_events']==10 and plan['maximum_cells']==60,'Fixed ten/sixty scope required')
 require([{k:v for k,v in e.items() if k!='locator_inventory'} for e in plan['events']]==[{k:v for k,v in e.items() if k!='locator_inventory'} for e in events],'Original event metadata changed')
 rows=[]
 for spec in plan['events']:
  raw=v1.safe.local(root,spec['source']['path']).read_bytes()
  require(v1.safe.digest(raw)==spec['source']['sha256'],'Source changed')
  rows.append(map_event(spec,raw,plan))

 previous=v1.read(root,'outputs/monthly_cashflow_pdf_repair_v2/RAW_FACTS.json',V2_PINS['outputs/monthly_cashflow_pdf_repair_v2/RAW_FACTS.json'])
 old={e['event_id']:e for e in previous['events']}
 preserved=0
 for event in rows:
  require(event['event_id'] in old,'Original v2 event missing')
  for side in v1.SIDES:
   for field in v1.FIELDS:
    prior=old[event['event_id']]['fields'][side][field]
    if prior is None:continue
    current=event['fields'][side][field]
    require(current is not None,'V3 cannot lose a recovered V2 value')
    for key in ('displayed_amount','value_decimal','scale','currency_literal','currency_iso','period','available_from','source'):
     require(current[key]==prior[key],'V3 changed recovered V2 field: '+key)
    preserved+=1
 require(preserved==previous['counts']['extracted_cells']==50,'Exact50 recovered V2 values required')
 count=sum(e['fields'][s][f] is not None for e in rows for s in v1.SIDES for f in v1.FIELDS)
 data=dict(schema_version=2,stage='raw_source_repair_candidates_not_admitted',plan_sha256=plan_sha,events=rows,
  counts=dict(events=10,requested_cells=60,extracted_cells=count,gap_cells=60-count),fact_admissions=0,model_admissions=0,strategy_admissions=0,
  financial_calculations=0,ratios_computed=0,signs_computed=0,ranks_computed=0,returns_computed=0,model_fits=0,
  first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
 audit=dict(schema_version=2,stage='source_repair_audit_not_admission',passed=True,
  meaning='Pinned inputs and deterministic extraction process verified; incomplete or ambiguous cells remain gaps, no source or model admission.',
  plan_sha256=plan_sha,data_sha256=v1.safe.digest(v1.safe.encoded(data)),counts=data['counts'],complete_cell_coverage=count==60,
  source_hashes=plan['source_hashes'],source_code_hashes=plan['source_code_hashes'],pdf_runtime_hashes=plan['pdf_runtime_hashes'],
  all_ten_original_events_preserved=True,all_rejected_prior_events_preserved=True,independent_new_field_review_pending=True,
  v1_frozen_artifacts_unchanged=True,v2_frozen_artifacts_unchanged=True,v2_values_preserved=preserved,financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0,fact_admissions=0)
 verified_inputs(root)
 return data,audit


def build(root,plan_sha,verify=False):
 data,audit=products(root,plan_sha);values={'RAW_FACTS.json':data,'AUDIT.json':audit}
 manifest=dict(plan_sha256=plan_sha,output_hashes={k:v1.safe.digest(v1.safe.encoded(v)) for k,v in values.items()})
 values['MANIFEST.json']=manifest
 for name,value in values.items():
  if verify:require(v1.read(root,OUT+'/'+name)==value,'Reproduced output mismatch')
  else:v1.publish(root,OUT+'/'+name,value)
 return dict(passed=True,verified=verify,counts=data['counts'],output_hashes=manifest['output_hashes'])


def main():
 parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
 for flag in ('draft-plan','freeze-plan','build','verify'):mode.add_argument('--'+flag,action='store_true')
 parser.add_argument('--draft-sha');parser.add_argument('--plan-sha');args=parser.parse_args()
 result=draft(ROOT) if args.draft_plan else freeze(ROOT,args.draft_sha) if args.freeze_plan else build(ROOT,args.plan_sha,args.verify)
 print(v1.safe.json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
