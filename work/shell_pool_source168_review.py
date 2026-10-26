"""Read-only original-PDF review of Shell168; never imports the producer."""
from pathlib import Path
import argparse,hashlib,io,json,re,sys
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
O=R/'outputs/monthly_shell_pool_source168'
PLAN_SHA='587fa9546ef99a84a947a8f51bb6e66b768454fdb9474faeefac9a2cf41687eb'
D=O/'review_render'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
 with Path(path).open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load_plan():
 assert sha(O/'PLAN.json')==PLAN_SHA
 plan=json.loads((O/'PLAN.json').read_text());assert len(plan['records'])==20
 for record in plan['records']:
  for rel,h in record['inputs'].items():assert sha(R/rel)==h,(rel,h)
 for rel,h in plan['anchors'].items():assert sha(R/rel)==h,(rel,h)
 return plan

def probe():
 plan=load_plan();D.mkdir(parents=True,exist_ok=True)
 sys.path.append('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
 from pypdf import PdfReader
 import pypdfium2 as pdfium
 result=[]
 for record in plan['records']:
  ident='%sQ%s'%(record['year'],record['quarter']);raw=(R/record['binary']).read_bytes()
  reader=PdfReader(io.BytesIO(raw));document=pdfium.PdfDocument(raw)
  rows=[]
  for index,page in enumerate(reader.pages):
   text=page.extract_text(extraction_mode='layout')
   normalized=' '.join(text.split())
   hit=bool(re.search(r'(?:SHARE CAPITAL|issued.{0,80}ordinary|ordinary.{0,100}issued|American Depositary|American depositary|ADS represents|ADS equivalent)',normalized,re.I))
   if hit or index==0:
    dest=D/(ident+'_%02d'%(index+1));dest.with_suffix('.txt').write_text(text)
    image=document[index].render(scale=1.2).to_pil();image.save(dest.with_suffix('.png'))
    independent_text=document[index].get_textpage().get_text_range()
    dest.with_suffix('.pdfium.txt').write_text(independent_text)
    lines=[line.strip() for line in text.splitlines() if re.search(r'(issued|treasury|trust|weighted|American|ADS|February|April|May|July|August|October|November|January)',line,re.I)]
    rows.append(dict(page=index+1,layout_text=str(dest.with_suffix('.txt').relative_to(R)),render=str(dest.with_suffix('.png').relative_to(R)),text_excerpt=lines))
  result.append(dict(id=ident,binary=record['binary'],binary_sha256=sha(R/record['binary']),page_count=len(reader.pages),pages=rows))
  print(json.dumps(dict(id=ident,pages=[r['page'] for r in rows],excerpts=rows),ensure_ascii=False))
 dump(D/'PROBE.json',dict(plan_sha256=PLAN_SHA,records=result,source_files_checked=20,created_utc=datetime.now(timezone.utc).isoformat()))

def dates():
 plan=load_plan();D.mkdir(parents=True,exist_ok=True)
 sys.path.append('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
 import pypdfium2 as pdfium
 from PIL import Image,ImageDraw
 result=[];crops=[]
 for record in plan['records']:
  ident='%sQ%s'%(record['year'],record['quarter']);document=pdfium.PdfDocument((R/record['binary']).read_bytes());hits=[];concepts=[]
  for index in range(len(document)):
   page=document[index];text=page.get_textpage().get_text_range()
   lines=text.splitlines()
   for j,line in enumerate(lines):
    if re.search(r'(each.{0,30}(ADS|depositary)|ADS.{0,40}represent|treasury|held in trust|weighted.average)',line,re.I):
     concepts.append(dict(page=index+1,text=' '.join(lines[max(0,j-1):j+3])))
   if index==len(document)-1 or re.search(r'The information in this (report|announcement)',text,re.I):
    dest=D/(ident+'_release_%02d'%(index+1));dest.with_suffix('.pdfium.txt').write_text(text)
    image=page.render(scale=1.4).to_pil();image.save(dest.with_suffix('.png'))
    crop=image.crop((0,max(0,image.height-620),image.width,image.height));crop=crop.convert('RGB')
    ImageDraw.Draw(crop).text((20,5),ident+' PDF page '+str(index+1),fill='red');crops.append(crop)
    hits.append(dict(page=index+1,render=str(dest.with_suffix('.png').relative_to(R)),text=str(dest.with_suffix('.pdfium.txt').relative_to(R)),tail=text[-2000:]))
  result.append(dict(period=ident,pages=hits,concepts=concepts));print(json.dumps(result[-1],ensure_ascii=False))
 for start in range(0,len(crops),4):
  group=crops[start:start+4];w=max(x.width for x in group);h=max(x.height for x in group);sheet=Image.new('RGB',(w*2,h*2),'white')
  for k,im in enumerate(group):sheet.paste(im,((k%2)*w,(k//2)*h))
  sheet.save(D/('release_contact_%02d.png'%(start//4+1)))
 dump(D/'DATE_PROBE.json',dict(plan_sha256=PLAN_SHA,records=result))

# Independent visual transcription of all twenty original capital tables
# (read before producer records) and announcement-date pages.
# Columns: period, capital PDF page, release PDF page, stated release date, A, B.
VISUAL='''2015Q4 16 19 2016-02-04 3990921569 2440410614
2016Q1 18 23 2016-05-04 4275353925 3745486731
2016Q2 18 30 2016-07-28 4325899655 3745486731
2016Q3 18 24 2016-11-01 4369954444 3745486731
2016Q4 20 27 2017-02-02 4428903813 3745486731
2017Q1 12 18 2017-05-04 4476695491 3745486731
2017Q2 13 23 2017-07-27 4510617762 3745486731
2017Q3 12 19 2017-11-02 4544414617 3745486731
2017Q4 14 21 2018-02-01 4597136050 3745486731
2018Q1 13 20 2018-04-26 4597136050 3745486731
2018Q2 14 25 2018-07-26 4597136050 3745486731
2018Q3 13 21 2018-11-01 4554081081 3745486731
2018Q4 15 23 2019-01-31 4471889296 3745486731
2019Q1 13 23 2019-05-02 4399358177 3745486731
2019Q2 14 28 2019-08-01 4332474849 3745486731
2019Q3 13 25 2019-10-31 4244662769 3733998448
2019Q4 15 27 2020-01-30 4151787517 3729407107
2020Q1 14 21 2020-04-30 4105643625 3713984248
2020Q2 15 27 2020-07-30 4101239499 3706183836
2020Q3 16 24 2020-10-29 4101239499 3706183836'''
OPENINGS={2015:(3907302393,2440410614),2016:(3990921569,2440410614),
 2017:(4428903813,3745486731),2018:(4597136050,3745486731),
 2019:(4471889296,3745486731),2020:(4151787517,3729407107)}
MOVEMENTS={
 '2015Q4':[(96336688,0),(-12717512,0)],
 '2016Q1':[(65704048,0),(218728308,1305076117)],
 '2016Q2':[(116249778,0),(218728308,1305076117)],
 '2016Q3':[(160304567,0),(218728308,1305076117)],
 '2016Q4':[(219253936,0),(218728308,1305076117)],
 '2017Q1':[(47791678,0)],'2017Q2':[(81713949,0)],
 '2017Q3':[(115510804,0)],'2017Q4':[(168232237,0)],
 '2018Q1':[],'2018Q2':[],'2018Q3':[(-43054969,0)],'2018Q4':[(-125246754,0)],
 '2019Q1':[(-72531119,0)],'2019Q2':[(-139414447,0)],
 '2019Q3':[(-227226527,-11488283)],'2019Q4':[(-320101779,-16079624)],
 '2020Q1':[(-46143892,-15422859)],'2020Q2':[(-50548018,-23223271)],
 '2020Q3':[(-50548018,-23223271)]}
UNRESOLVED=[
 'Contemporaneous public-availability timestamp and original historical PDF version are not independently proven by the later retrieval.',
 'Issued capital is not verified class-specific net outstanding: treasury and employee-trust A/B counts remain unresolved.',
 'The twenty QRA PDFs do not establish a complete dated class-specific ADS-to-ordinary conversion history.',
 'Class-specific unadjusted historical quotes, currency/quote-unit and original vendor market-cap coverage remain outside this evidence set.',
 'Quarter-end capital observations do not prove intervening daily share counts or Top32/Top35 membership.']

def review():
 plan=load_plan();extraction_path=O/'EXTRACTION.json';extraction_sha=sha(extraction_path)
 producer=json.loads(extraction_path.read_text());assert producer['plan_sha256']==PLAN_SHA
 rows={x['period']:x for x in producer['records']};assert len(rows)==20
 sys.path.append('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
 import pypdfium2 as pdfium
 visual={line.split()[0]:line.split()[1:] for line in VISUAL.splitlines()}
 records=[];mismatches=[]
 for source in plan['records']:
  ident='%sQ%s'%(source['year'],source['quarter']);cp,rp,release,a,b=visual[ident]
  cp,rp,a,b=map(int,(cp,rp,a,b));year=source['year'];q=source['quarter']
  end='%04d-%02d-%02d'%(year,q*3,{1:31,2:30,3:30,4:31}[q])
  doc=pdfium.PdfDocument((R/source['binary']).read_bytes())
  pages=[' '.join(doc[i].get_textpage().get_text_range().split()) for i in range(len(doc))]
  capital=pages[cp-1];release_text=pages[rp-1]
  date_string=datetime.fromisoformat(release).strftime('%B %d, %Y').replace(' 0',' ')
  assert date_string in release_text,(ident,'release')
  assert all(format(value,',') in capital for value in (a,b)),(ident,'counts')
  end_string=datetime.fromisoformat(end).strftime('%B %d, %Y').replace(' 0',' ')
  assert end_string in capital,(ident,'period_end')
  assert re.search(r'issued.{0,30}(?:fully|share)',capital,re.I),(ident,'issued basis')
  p=rows[ident];s=p['issued_ordinary_shares'];rel=p['release'];roll=p['rollforward'];op=OPENINGS[year]
  manual_moves=MOVEMENTS[ident]
  assert op[0]+sum(x[0] for x in manual_moves)==a and op[1]+sum(x[1] for x in manual_moves)==b
  checks={
   'A':s['A']==a,'B':s['B']==b,'period_end':p['period_end']==end,
   'capital_pdf_page':s['source_page_1based']==cp,'release_pdf_page':rel['source_page_1based']==rp,
   'release_stated_date':rel['stated_date']==release,'sum_labeled_derived':s['total_derived']==a+b,
   'ordinary_eur_nominal':s['nominal_per_ordinary_share_eur']=='0.07',
   'opening_counts':(roll['A_open'],roll['B_open'])==op,
   'opening_date':roll['opening_date']=='%04d-01-01'%year,
   'movements':sorted((x['A'],x['B']) for x in roll['movements'] if x['A'] or x['B'])==sorted(manual_moves),
   'rollforward':roll['computed_A_end']==a and roll['computed_B_end']==b and roll['agrees'] is True,
   'binary_sha256':p['source']['sha256']==sha(R/source['binary']),
   'source_identity':p['source']['binary']==source['binary'] and p['source']['url']==source['url'],
   'availability_not_overclaimed':rel['historical_availability_proven'] is False and rel['intraday_time'] is None,
   'not_pool_admission':p['strategy_admitted'] is False and p['pool_modified'] is False,
  }
  findings=[dict(field=k,problem='producer differs from independent source transcription') for k,v in checks.items() if not v]
  if findings:mismatches.append(dict(period=ident,findings=findings))
  count_render=D/(ident+'_%02d.png'%cp);date_render=D/(ident+'_release_%02d.png'%rp)
  assert count_render.is_file() and date_render.is_file()
  trust_pages=[i+1 for i,t in enumerate(pages) if re.search(r'shares held in trust',t,re.I)]
  eps_pages=[i+1 for i,t in enumerate(pages) if re.search(r'weighted average number of shares',t,re.I)]
  ads_pages=[i+1 for i,t in enumerate(pages) if re.search(r'\bADS\b|American Depositary',t,re.I)]
  ratio_hits=[]
  for i,t in enumerate(pages):
   ratio_hits.extend(dict(page=i+1,text=m.group(0)) for m in re.finditer(r'.{0,50}(?:each\s+(?:American\s+Depositary\s+Share|ADS).{0,100}(?:represent|ordinary)|ADS.{0,50}represents?\s+(?:two|2)).{0,80}',t,re.I))
  flags=[]
  if ident=='2017Q3':flags.append('Bold PDF text is duplicated by extraction; original rendered A/B columns were read visually, not guessed from rollforward.')
  if ident=='2020Q2':flags.append('Physical capital PDF page 15 prints Page 16; physical release page 27 prints Page 28. Physical page references are intentional.')
  records.append(dict(period=ident,source_binary=source['binary'],source_url=source['url'],source_sha256=sha(R/source['binary']),
   period_end=end,independent_A=a,independent_B=b,capital_pdf_page_1based=cp,release_stated_date=release,release_pdf_page_1based=rp,
   visual_review_completed=True,capital_render=str(count_render.relative_to(R)),capital_render_sha256=sha(count_render),
   release_render=str(date_render.relative_to(R)),release_render_sha256=sha(date_render),checks=checks,mismatches=findings,
   basis='Issued and fully paid euro 0.07 ordinary A/B shares; exact individual shares, not monetary nominal values, ADS units or EPS averages.',
   conceptual_checks=dict(trust_disclosure_pages=trust_pages,weighted_EPS_pages=eps_pages,ADS_word_pages=ads_pages,
    explicit_ADS_ratio_matches=ratio_hits,deferred_shares_excluded=50000,issued_equals_outstanding_proven=False,ADS_ratio_history_proven=False),
   extraction_flags=flags,unresolved=list(UNRESOLVED)))
 assert sha(extraction_path)==extraction_sha,'producer changed during review'
 out=dict(schema='shell_pool_source168_independent_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
  plan_sha256=PLAN_SHA,reviewer_source_sha256=sha(__file__),producer_extraction_sha256=extraction_sha,
  method='All 20 count tables manually read from original PDF renders; all 20 stated release dates visually read on original date pages. PDFium source extraction corroborates the transcription; pypdf was used only to locate render pages. Producer code was not imported.',
  source_scope='Twenty PLAN-bound local issuer PDFs, 2015Q4 through 2020Q3; read-only offline source audit.',
  summary=dict(documents_reviewed=len(records),source_input_hashes_checked=sum(len(x['inputs']) for x in plan['records']),
   anchor_hashes_checked=len(plan['anchors']),count_pairs_matched=sum(x['checks']['A'] and x['checks']['B'] for x in records),
   dates_matched=sum(x['checks']['period_end'] and x['checks']['release_stated_date'] for x in records),
   rollforwards_matched=sum(x['checks']['movements'] and x['checks']['rollforward'] for x in records),
   extraction_mismatch_count=len(mismatches),unresolved_capital_row_transcriptions=0,
   conceptual_limitations_remain=True,overall='PASS_SOURCE_TRANSCRIPTION_ONLY' if not mismatches else 'FAIL_SOURCE_MISMATCH'),
  records=records,mismatches=mismatches,
  conceptual_findings=[
   'Treasury/trust monetary balances and EPS weighted-average counts cannot be substituted for date-specific A/B outstanding counts.',
   'The 50,000 sterling deferred shares are a separate security, excluded from A/B ordinary-share counts.',
   'Announcement dates stated inside later retrieved PDFs are useful evidence but do not establish contemporaneous bytes, publication time or vintage access.',
   'The prior 2008/2009 SHELL_CLASS_EVIDENCE anchor reports a two-ordinary-per-ADS convention and uses outstanding counts; this audit does not extend that ratio or basis to 2015-2020 without new evidence.',
   'No issuer capitalization, rank, pool eligibility or strategy outcome follows directly from this audit.'],
  unresolved=list(UNRESOLVED),external_audit_review=external_review(),network_requests=0,strategy_trials_added=0,pool_modified=False,market_caps_computed=False,observer_modified=False)
 dump(O/'INDEPENDENT_REVIEW.json',out);print(json.dumps(out['summary'],sort_keys=True))

def external_review():
 """Read the parent's four saved receipts/bodies, without requesting any URL."""
 from html import unescape
 audit_path=O/'EXTERNAL_AUDIT.json';audit=json.loads(audit_path.read_text())
 collection_path=O/'external/COLLECTION.json';collection=json.loads(collection_path.read_text())
 assert sha(collection_path)==audit['collection_sha256']
 bodies={};receipt_checks=[]
 for r in collection['records']:
  path=R/r['path'];receipt_path=O/'external'/(r['name']+'.receipt.json')
  receipt=json.loads(receipt_path.read_text());assert receipt==r
  assert len(path.read_bytes())==r['bytes'] and sha(path)==r['sha256']
  bodies[r['name']]=path.read_text()
  receipt_checks.append(dict(name=r['name'],status=r['status'],source_sha256=r['sha256'],receipt_sha256=sha(receipt_path),body_matches_receipt=True))
 assert len(bodies)==4
 wire=bodies['issuer_wire_20160729_capital'];plain=' '.join(unescape(re.sub('<[^>]+>',' ',wire)).split())
 w=audit['issuer_wire']
 checks={
  'wire_timestamp':w['stated_publication']=='2016-07-29T07:22:00-04:00' and w['stated_publication'] in wire,
  'wire_A':w['class_A_issued']==4325899655 and '4,325,899,655 A shares' in plain,
  'wire_B':w['class_B_issued']==3745486731 and '3,745,486,731 B shares' in plain,
  'wire_total':w['total_issued']==8071386386 and '8,071,386,386' in plain,
  'wire_zero_treasury':w['treasury_ordinary_shares']==0 and 'holds no ordinary shares in Treasury' in plain,
  'wire_includes_uncancelled':bool(re.search('include shares purchased.{0,140}but not yet cancelled',plain)),
  'wire_not_historical_vintage':w['exact_historical_retrieval_vintage_verified'] is False,
  'wire_source_hash':w['source_sha256']==sha(R/'outputs/monthly_shell_pool_source168/external/issuer_wire_20160729_capital.html'),
  'vendor_embedded_history':bool(re.search(r'data\s*=\s*\[\{"d":\d+,"m":\d+',bodies['current_vendor_history'])),
  'vendor_basis_not_asserted':audit['vendor']['old_single_class_basis_verified'] is False,
  'ratio_body_not_evidence':audit['issuer_ratio_page']['explicit_ratio_in_retrieved_body'] is False and not re.search(r'\bADS\b|depositary|ordinary shares',bodies['issuer_2016q2_ads_ratio'],re.I),
  'ratio_receipt_status':audit['issuer_ratio_page']['status']==next(x['status'] for x in collection['records'] if x['name']=='issuer_2016q2_ads_ratio')==200,
  'sec_block_body':audit['sec']['status']==403 and 'Your Request Originates from an Undeclared Automated Tool' in bodies['sec_20160919_issuance'],
  'sec_not_admitted':audit['sec']['body_admitted'] is False,
 }
 manifest=json.loads((R/'outputs/monthly_top35_extension/CAP_COLLECTION.json').read_text())
 old=next(x for x in manifest['records'] if x['symbol']=='SHEL');missing=[]
 for entry,field in zip(audit['original_vendor_files'],['raw_sha256','sha256']):
  p=Path(entry['path']);assert p.is_relative_to(R),'no OneDrive access'
  checks['missing_'+p.name]=(p.exists() is False and entry['exists'] is False and entry['expected_sha256']==old[field])
  missing.append(dict(path=str(p.relative_to(R)),expected_sha256=old[field],exists=p.exists()))
 checks['new_vendor_bytes_differ']=audit['vendor']['source_sha256']==sha(R/'outputs/monthly_shell_pool_source168/external/current_vendor_history.html') and audit['vendor']['source_sha256']!=old['raw_sha256'] and audit['vendor']['matches_original_raw_sha256'] is False
 assert all(checks.values()),checks
 return dict(status='PASS_SAVED_BODY_AND_RECEIPT_REVIEW',audit_sha256=sha(audit_path),collection_sha256=sha(collection_path),
  scope='Separate read-only review requested by parent; these four saved web bodies are not additions to the frozen twenty-PDF scope.',
  checks=checks,receipt_checks=receipt_checks,original_paths=missing,
  caveat='Only the two specified local original paths are checked for absence. No claim that every backup or remote archive was searched. No network request or market-cap calculation performed.',
  findings=['Wire body states A/B issued counts, total, zero treasury and uncancelled repurchase inclusion at its stated July 29 publication; this does not certify net shares excluding employee trusts.',
   'Current vendor HTML contains an embedded history array but differs from the missing original raw SHA, so its original historical basis is unresolved.',
   'Shell dividend URL returns an application shell without an explicit ADS ratio; HTTP 200 is not ratio evidence.',
   'SEC body is an automated-tool block page, consistent with its 403 receipt and non-admission.'])

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['probe','dates','review']);args=ap.parse_args();globals()[args.phase]()
