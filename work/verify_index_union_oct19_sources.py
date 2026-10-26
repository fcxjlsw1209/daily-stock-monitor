"""Offline, source-only verification; does not alter admission or strategies."""
import datetime, hashlib, json, re
from html.parser import HTMLParser
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/index_union/sources/oct19_dtcc_symbol_bridge'
class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.parts=[]
    def handle_data(self, data): self.parts.append(data)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_once(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
checks=[]; sources={}; texts={}
for name in ['issuer_20141006_symbol_change.html','malone_20161114_sc13da.html']:
    receipt=json.loads((OUT/(name+'.receipt.json')).read_text())
    raw=(OUT/name).read_bytes()
    for label, passed in [('http_200',receipt['http_status']==200),('source_digest',digest(OUT/name)==receipt['sha256']),('source_size',len(raw)==receipt['bytes'])]:
        if not passed: raise ValueError(name+': '+label)
        checks.append(name+': '+label)
    parser=Text();parser.feed(raw.decode('utf-8'))
    texts[name]=' '.join(' '.join(parser.parts).split()).replace('“','"').replace('”','"').replace(chr(147),'"').replace(chr(148),'"')
    sources[name]=receipt
release=texts['issuer_20141006_symbol_change.html']
filing=texts['malone_20161114_sc13da.html']
requirements=[
 ('release_date',release,r'Released October 6, 2014'),
 ('effective_market_open',release,r'effective at the market open on October 7, 2014'),
 ('series_A_transition',release,r'"LINTA".*?Series A.*?"QVCA"'),
 ('series_B_transition',release,r'"LINTB".*?Series B.*?"QVCB"'),
 ('series_A_cusip',filing,r'Series A QVC Group common stock: 53071M104'),
 ('series_B_cusip',filing,r'Series B QVC Group common stock: 53071M203'),
 ('ticker_cusip_pair_A',filing,r'53071M104 \(QVCA\)'),
 ('ticker_cusip_pair_B',filing,r'53071M203 \(QVCB\)'),
 ('signature_date',filing,r'November 14, 2016'),
 ('event_date',filing,r'November 4, 2016'),
 ('retrospective_transition',filing,r'"LINTA" was changed to "QVCA".*?"LINTB" trading symbol to "QVCB," effective October 7, 2014'),
 ('reporting_person',filing,r'John C\. Malone'),
]
locators={}
for label,text,pattern in requirements:
    match=re.search(pattern,text)
    if not match:raise ValueError('Missing literal source context: '+label)
    checks.append(label)
    locators[label]={'normalized_text_start':match.start(),'normalized_text_end':match.end(),'matching_text':match.group()}
failed_downloads=[]
for file,header,url in [
 ('otc193.download','response_headers.txt','https://www.dtcc.com/globals/pdfs/2014/october/07/otc-193'),
 ('otc193_legacy.download','legacy_response_headers.txt','https://www.dtcc.com/-/media/Files/pdf/2014/10/7/OTC--193.pdf')]:
    h=(OUT/header).read_text();codes=re.findall(r'HTTP/\S+\s+(\d{3})',h)
    if not codes or codes[-1]!='404':raise ValueError('Unexpected DTCC receipt')
    if (OUT/file).read_bytes().startswith(b'%PDF'):raise ValueError('Unexpected PDF under failed receipt')
    checks.append(file+': 404 HTML explicitly excluded')
    failed_downloads.append({'url':url,'body':file,'headers':header,'http_status':404,'body_sha256':digest(OUT/file),'headers_sha256':digest(OUT/header),'original_pdf_obtained':False,'admitted':False})
protected=['outputs/index_union/STATUS.json','outputs/index_union/BACKTEST_READINESS.json','outputs/index_union/NDX_EVENT_ADJUDICATION.json','outputs/index_union/AVAILABLE_RESEARCH_CODE_FINGERPRINTS.json','outputs/index_union/AVAILABLE_RESEARCH_ASSUMPTIONS.json','outputs/index_union/SECURITY_ALIAS_EVIDENCE_03.json']
protected_hashes={p:digest(ROOT/p) for p in protected}
facts={
 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Additional original source corroboration only; no change to admitted historical inputs',
 'sources':sources,'source_checks_passed':len(checks),'checks':checks,'locators':locators,
 'supported_facts':{
   'dated_issuer_release':{'publication_date_label':'2014-10-06','effective_market_open_date':'2014-10-07','series_A_old':'LINTA','series_A_new':'QVCA','series_B_old':'LINTB','series_B_new':'QVCB','previously_known_alias_recovered_as_original_bytes':True},
   'later_reporting_person_filing':{'reporting_person':'John C. Malone','form':'Schedule 13D/A amendment 4','event_date':'2016-11-04','signature_date':'2016-11-14','CUSIP_QVCA':'53071M104','CUSIP_QVCB':'53071M203','date_of_publication_or_SEC_acceptance_verified':False,'2014_change_retrospectively_recounted':True}},
 'limitations':['2016 CUSIP mapping does not certify unchanged identifiers on October 7, 2014.','DTCC PDF original bytes and table images remain unavailable; web extracts are discovery leads only.','Neither document certifies full permanent identity history, complete corporate action returns, cash credit dates, execution prices, adjusted/raw units or index membership.','Later filing is not backdated into 2014 decision inputs.'],
 'failed_downloads':failed_downloads,'strict_admission_changed':False,'exclusions_changed':False,'backtests_rerun':False,'observers_accessed':False,
 'separate_existing_regression_checks':{'security_aliases':{'count':4,'passed':True},'research_scope':{'count':4,'passed':True},'evidence':'Root tool executions on this pass both exited 0; no new strategy tests or full suite rerun.'},
 'protected_inputs_sha256':protected_hashes,'six_membership_disputes_remain':True,
 'further_work':'Obtain contemporaneous CUSIP transition original; reconcile remaining membership and corporate-action evidence without weakening strict gates.'}
write_once('SOURCE_REVIEW.json',facts)
write_once('MANIFEST.json',{'files':{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file()},'verifier':{'path':'work/verify_index_union_oct19_sources.py','sha256':digest(Path(__file__))}})
print(json.dumps({'source_checks_passed':len(checks),'original_source_downloads':len(sources),'dtcc_originals_available':False,'strict_admission_changed':False,'manifest_sha256':digest(OUT/'MANIFEST.json')}))
