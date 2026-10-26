"""Amount-free scope addendum for the completed CSCO DOM context review."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,re
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work/after149_csco_source_context.json'
OLD='657b3816cbf4794f66dccdd0b449dce1849f4038be65e89a36751dca33aeb55b'
def sha(b):return hashlib.sha256(b).hexdigest()
b=P.read_bytes();assert sha(b)==OLD
x=json.loads(b)
for p,h in x['source_hashes'].items():assert sha((ROOT/p).read_bytes())==h,p
notes=[]
for e in x['events']:
 if e['fiscal_year'] not in (2018,2019):continue
 raw=(ROOT/e['source']['path']).read_bytes();soup=BeautifulSoup(raw,'html.parser')
 for n in soup(['script','style']):n.decompose()
 text=re.sub(r'\s+',' ',soup.get_text(' ',strip=True))
 seen=set()
 for m in re.finditer(r'ASC\s*606|ASU\s*2016-18|modified retrospective|new revenue recognition',text,re.I):
  start=max(text.rfind('. ',0,m.start()),text.rfind(': ',0,m.start()))+2
  if start<0 or m.start()-start>250:start=max(0,m.start()-200)
  end=text.find('. ',m.end());end=min(end+1 if end!=-1 else m.end()+400,m.end()+700)
  piece=text[start:end]
  redacted=re.sub(r'\d[\d,.]*','[number]',piece)
  if redacted in seen:continue
  seen.add(redacted)
  notes.append(dict(event_id=e['event_id'],source=e['source']['path'],source_sha256=e['source']['sha256'],text_position=m.start(),fiscal_year_references=re.findall(r'fiscal\s+(20\d{2})',piece,re.I),standard_references=re.findall(r'(?:ASC|ASU)\s*[0-9-]+',piece,re.I),redacted_context=redacted))
x['field_level_comparability']={
 'classification_is_not_admission':True,
 'NI_and_assets_vs_separate_CFO_tax_rows':{
  'years':list(range(2011,2019)),
  'status':'no_automatic_NI_or_assets_hold_from_CFO_tax_line_alone',
  'reason':'The separate excess-tax-benefits line is below the consolidated net-income starting line in the operating reconciliation and also appears in financing. Its presence/classification is not itself evidence that the reported net-income starting value or year-end total assets uses a different perimeter. This does not assert that tax accounting has no effect on GAAP earnings or assets.',
  'limits':'Physical value checks, annual boundaries, dollar units and report-specific accounting policies remain unadmitted. Do not modify NI by the tax line; do not silently net tax benefits with SBC.'},
 'FY2018_annual_vs_future_guidance':{
  'status':'forward_guidance_ASC606_note_does_not_prove_FY2018_columns_recast',
  'reason':'The FY2018 release explicitly dates modified-retrospective ASC606 adoption to the beginning of fiscal2019 and describes forward revenue guidance. Preserve this as a FY2019 comparability issue, not an automatic statement that FY2018 annual cells were restated.'},
 'FY2019_current_and_comparison_assets':{
  'status':'hold_comparability_until_transition_scope_verified_not_a_missing_cell',
  'reason':'FY2019 current and prior asset column locators are present. The preceding FY2018 original release documents ASC606 modified-retrospective adoption at the beginning of FY2019. This gives a concrete possible recognition-basis break between FY2018 closing assets and FY2019 current figures. The FY2019 restricted-cash note confirms a different cash-flow presentation change; that note alone cannot clear the asset or earnings transition scope.',
  'needed_evidence':'Original annual-report transition note or similarly original-dated evidence stating whether prior-period balance-sheet/earnings columns were recast and how the opening FY2019 cumulative adjustment was recorded. No zero-impact or recast assumption.',
  'allowed_next_step':'Source-only extraction with transition flags retained. No automatic comparative-factor admission.'},
 'FY2019_cashflow_restricted_cash':{
  'status':'retain_separate_comparison_flow_scope_flag',
  'reason':'The source explicitly says prior-period information was retrospectively adjusted for ASU2016-18 Restricted Cash. Do not equate this with evidence that net cash from operations, net income or total assets numerically changed; no cross-report values were compared.'}
}
x['additional_redacted_accounting_context']=notes
x['source_hashes']['work/after149_csco_accounting_context_check.py']=sha(Path(__file__).read_bytes())
x['scope_addendum_created_utc']=datetime.now(timezone.utc).isoformat()
x['prior_context_record_sha256']=OLD
Path('/tmp/after149_csco_source_context_before_scope_addendum.json').write_bytes(b)
for p,h in x['source_hashes'].items():assert sha((ROOT/p).read_bytes())==h,p
P.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(dict(passed=True,path=str(P),sha256=sha(P.read_bytes()),amounts_extracted=0,additional_redacted_context=notes),indent=2))
