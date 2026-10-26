"""One SAP training predecessor; offline build from explicitly tiered evidence."""
from source_io import *
import sys
sys.path.insert(0,str(HERE.parent.parent))
from event_schema import normalize_event,event_available

PDF=HERE/'SAP/raw/6ae054fcd42fa322d88e.bin'
PUBLICATION=HERE/'PUBLICATION_CORROBORATION.json'

def clean(s):return re.sub(r'\s+',' ',s).strip()

def parse(pages,publication):
    if publication.get('headline')!='SAP veröffentlicht Rekordergebnisse für das dritte Quartal':
        raise ValueError('Actual completed-results issuer announcement required')
    dm=re.fullmatch(r'\(PresseBox\) \(Walldorf, (\d{2}\.\d{2}\.\d{4})\)',publication.get('dateline',''))
    if not dm:raise ValueError('Independent issuer-issued original dateline required')
    release=dt.datetime.strptime(dm.group(1),'%d.%m.%Y').date().isoformat()
    qm=re.search(r'im (\d)\. Quartal (20\d{2})',publication.get('quarter_excerpt',''))
    if not qm:raise ValueError('Issuer announcement fiscal identity absent')
    q,fy=map(int,qm.groups())
    if (fy,q)!=(2012,3):raise ValueError('Outside assigned SAP 2012Q3')
    p25=clean(pages[24]['text'])
    pm=re.search(r'CONSOLIDATED INCOME STATEMENTS For the three months ended (\w+ \d+) € millions, unless otherwise stated Note (20\d{2}) (20\d{2})',p25)
    if not pm:raise ValueError('Actual current three-month period table absent')
    if int(pm.group(2))!=fy or int(pm.group(3))!=fy-1:raise ValueError('Current versus comparative fiscal year conflict')
    end=dt.datetime.strptime(pm.group(1)+' '+pm.group(2),'%B %d %Y').date().isoformat()
    p54=clean(pages[53]['text'])
    im=re.search(r'Imprint Overall Responsibility: SAP AG Corporate Financial Reporting Published on (\w+ \d+, \d{4})',p54)
    if not im:raise ValueError('Issuer-authored report imprint absent')
    if dt.datetime.strptime(im.group(1),'%B %d, %Y').date().isoformat()!=release:
        raise ValueError('Report publication and independently issued results dates conflict')
    if end!='2012-09-30' or not end<release:raise ValueError('Target actual end invalid')
    return dict(release_date=release,fiscal_year=fy,fiscal_quarter=q,fiscal_period_end_actual=end,period_excerpt=pm.group(),imprint_excerpt=im.group())

def normalized(e):
    return normalize_event(dict(symbol=e['ticker'],official=dict(verified=True,publication_kind='issuer_release',publication_date=e['release_date'],publication_timezone=e['publication_timezone'],actual_period_end=e['fiscal_period_end_actual'],source_url=e['source_url'])))

def build():
    raw=PDF.read_bytes();assert raw.startswith(b'%PDF')
    meta=json.loads(PDF.with_suffix('.meta.json').read_text());assert sha(raw)==meta['sha256'] and meta['status']==200
    pp=PDF.with_suffix('.pages.json');tm=json.loads(PDF.with_suffix('.text.meta.json').read_text())
    assert sha(raw)==tm['source_sha256'] and sha(pp.read_bytes())==tm['extracted_pages_sha256']
    pages=json.loads(pp.read_text());publication=json.loads(PUBLICATION.read_text());r=parse(pages,publication)
    primary=dict(evidence_path=rel(PDF),sha256=sha(raw),source_url=meta['url'])
    wire=dict(evidence_path=rel(PUBLICATION),sha256=sha(PUBLICATION.read_bytes()),source_url=publication['source_url'],source_class=publication['source_class'])
    e=dict(ticker='SAP',**{k:r[k] for k in ['release_date','fiscal_year','fiscal_quarter','fiscal_period_end_actual']},publication_timestamp=None,precision='date_only',publication_timezone='Europe/Berlin',publication_timezone_basis='Issuer-local Walldorf date; no publication clock observed; central date-only next-close policy applies',first_public_time_verified=False,source_url=meta['url'],source_class='issuer_authored_interim_report_pdf_full_bytes_third_party_mirror',evidence_path=rel(PDF),evidence_sha256=sha(raw),source_retrieved_at=meta['retrieved_at'],download_host_is_official=False,full_original_format_pdf_bytes=True,issuer_at_event='SAP AG',issuer_current='SAP SE',asset_identity='SAP NYSE ADR, each representing one common share; issuer report PDF page20 identifies ADR and ticker',release_date_basis='Actual issuer-issued PresseBox completed-results dateline, independently matching the issuer-authored report imprint; report imprint alone is not used as earliest earnings publication',release_date_evidence=[dict(wire,excerpt=publication['headline']+'; '+publication['dateline']+'; '+publication['quarter_excerpt']),dict(primary,pdf_page=54,excerpt=r['imprint_excerpt'],role='corroboration_only_not_standalone_earliest_release_date')],fiscal_identity_evidence=dict(wire,excerpt=publication['quarter_excerpt']),fiscal_period_evidence=dict(primary,pdf_page=25,excerpt=r['period_excerpt'],basis='Current-year 2012 column in explicit three months ended September30 income statement; comparative2011 excluded'),corroborating_public_index_evidence=wire,discovery_source_url='https://stockanalysis.com/filings/ETR:SAP/2037597/',historical_limitation='PDF is issuer-authored but downloaded from Quartr public mirror, not SAP host. Independent issuer-issued announcement is a public-index excerpt, not full original HTML. No precise or earliest-ever financial-disclosure clock claim.',event_scope='Regular quarterly event date and fiscal metadata only; no EPS or revised financial values imported',training_predecessor_only=True,in_study_window=False,scope_reason='Required prior actual earnings event for SAP training candidate2012-12-31; no unrelated historical expansion')
    # Central adapter hashes additional_evidence and preserves the explicit
    # limitation/identity fields. Retain these alongside richer source notes.
    e['additional_evidence']=[dict(wire,role='independent_issuer_issued_actual_release_date')]
    e['evidence_limitation']=e['historical_limitation']+' Publication date comes from the actual issuer-issued completed-results dateline; the matching report imprint only corroborates it. Issuer-local Berlin date-only conservative availability applies.'
    e['issuer_identity_note']='SAP AG at the2012 event, subsequently SAP SE; NYSE SAP ADR represents one common share per report page20. Quartr is the third-party PDF mirror, not the issuer.'
    dump(HERE/'events.json',[e])
    inventory=[]
    for mp in sorted((HERE/'SAP/raw').glob('*.meta.json')):
        if '.text.' in mp.name:continue
        m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
        inventory.append(dict(source_url=m['url'],evidence_path=rel(p),sha256=sha(p.read_bytes()),status=m['status'],role='admitted_report' if p==PDF else 'discovery_or_unadmitted',retry_after=m.get('retry_after')))
    dump(HERE/'SOURCE_INVENTORY.json',inventory)
    dump(HERE/'PENDING.json',dict(missing_assigned_quarters=[],pending_enrichment=['Original-host report or full issuer press-release HTML remains unavailable; transparency retained and no first-publication clock inferred'],failed_and_unused_sources_retained=True,source_stops_path=rel(HERE/'SOURCE_STOPS.json'),scope='Only actual training2012-12-31 predecessor; no full SAP archive claim'))
    print('SAP2012Q3',e['release_date'],e['fiscal_period_end_actual'],sha((HERE/'events.json').read_bytes()))

if __name__=='__main__':build()
