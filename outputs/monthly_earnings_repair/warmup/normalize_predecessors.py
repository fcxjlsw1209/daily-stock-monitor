"""Verify real prior-quarter releases for the first 2011 training month."""
import hashlib,json
from pathlib import Path
from bs4 import BeautifulSoup
from budvz_io import P,dump
OUT=P/'predecessors'
def main():
    p=OUT/'BRK_2010Q3.json';x=json.loads(p.read_text())
    assert x['publication_date']=='2010-11-05' and x['period_end']=='2010-09-30'
    common=dict(publication_timestamp=None,precision='date_only',publication_timezone='America/Chicago',
        first_public_time_verified=False,publication_version='current_archive',training_predecessor_only=True)
    rows=[dict(common,ticker='BRK.B',fiscal_year=2010,fiscal_quarter=3,release_date=x['publication_date'],fiscal_period_end_actual=x['period_end'],
        source_url=x['source_url'],evidence_path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),retrieved_at=x['retrieved_at'],
        source_kind='public_index_observation_of_original_release',period_basis=x['period_verification'],
        evidence_limitation='Public-index observation of original release and period only; original PDF bytes unavailable.')]
    for meta in (OUT/'raw').glob('*.json'):
        m=json.loads(meta.read_text())
        if m['url']!='https://stock.walmart.com/sec-filings/all-sec-filings/content/0001193125-10-261366/dex991.htm':continue
        p=meta.with_suffix('.bin');assert m['http_status']==200 and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
        text=' '.join(BeautifulSoup(p.read_bytes(),'html.parser').stripped_strings)
        for phrase in ('Nov. 16, 2010','quarter ended Oct. 31, 2010','third quarter of fiscal year 2011'):assert phrase in text,phrase
        rows.append(dict(common,ticker='WMT',fiscal_year=2011,fiscal_quarter=3,release_date='2010-11-16',fiscal_period_end_actual='2010-10-31',
            source_url=m['url'],evidence_path=str(p.resolve()),sha256=m['sha256'],retrieved_at=m['retrieved_at'],
            source_kind='issuer_hosted_original_release_exhibit',period_basis='Original release dateline and opening quarter-ended statement; not same-store retail calendar.'))
    assert len(rows)==2
    dump(OUT/'events.json',rows);print('Verified 2 training predecessors')
if __name__=='__main__':main()
