"""Record visual transcription of the four PTR class-count tables."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ptr_pool_source169/annual'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((OUT/'PLAN.json').read_text())
    extraction=json.loads((OUT/'EXTRACTION.json').read_text())
    assert extraction['plan_sha256']==sha(OUT/'PLAN.json')
    assert extraction['code_sha256']==sha(ROOT/'work/ptr169_annual_sources.py')==plan['code_sha256']
    secpath=ROOT/'outputs/dynamic_top100/priority_sec/PTR_submissions.json'
    assert sha(secpath)==plan['input_sha256'][str(secpath.relative_to(ROOT))]
    sec=json.loads(secpath.read_text())['filings']['recent']
    rows=[];pins={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'PLAN.json',OUT/'EXTRACTION.json',secpath]}
    for year,record in extraction['reports'].items():
        report=OUT/(year+'.pdf');receipt=OUT/(year+'.receipt.json')
        assert sha(report)==record['source_sha256']
        assert sha(receipt)==record['receipt_sha256']
        meta=json.loads(receipt.read_text())
        assert meta['body_sha256']==sha(report) and meta['status']==200 and meta['url']==plan['sources'][year]
        for p in [report,receipt]:pins[str(p.relative_to(ROOT))]=sha(p)
        page,=record['pages']
        for typ in ('text','image'):
            p=ROOT/page[typ]
            assert sha(p)==page[typ+'_sha256'];pins[page[typ]]=sha(p)
        text=(ROOT/page['text']).read_text()
        for expected in [year+' ANNUAL REPORT','161,922,077,818','21,098,900,000','183,020,977,818',
                         'Pre-movement','Post-movement','there was no issue of shares']:
            assert expected in text, (year,expected)
        matches=[i for i,form in enumerate(sec['form']) if form=='20-F' and sec['reportDate'][i]==year+'-12-31']
        assert len(matches)==1,(year,matches)
        i=matches[0]
        rows.append(dict(period_end=year+'-12-31',A_shares=161922077818,H_shares=21098900000,total=183020977818,
                         pre_movement_equals_post_movement=True,reported_new_share_issue=False,
                         unit='Shares; not millions of RMB or weighted-average EPS denominator',
                         exact_table_visual_review=True,source_page=page['pdf_page'],printed_page='009',
                         source=report.name,sec_index_filing_date=sec['filingDate'][i],
                         sec_index_accession=sec['accessionNumber'][i],
                         sec_index_primary_document=sec['primaryDocument'][i],
                         publication_time_verified_for_this_PDF=False))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='VISUAL_TRANSCRIPTION_VERIFIED_RELEASE_TIMING_SEPARATE',
                scope='Four annual class-count tables independently downloaded from exchange-hosted issuer reports.',
                visual_reviewer='root',visual_method='Viewed every existing full-page PNG, confirmed A/H and total rows, unit, pre/post columns and no-issue sentence.',
                rows=rows,input_sha256=pins,code_sha256=sha(Path(__file__)),
                findings=['All four reports give the same exact 2016-2019 A/H period-end counts as the 2015 report.',
                          'The reports state there was no new share issue in each reporting period.',
                          'A/H counts are company shares; ADSs must not be added as a third share class.'],
                limits=['A later annual report is not a contemporaneous announcement of every day in that year.',
                        'SEC filing dates index different 20-F documents; these dates are not certified publication timestamps of these HKEX PDFs.',
                        'A/H sum-of-local-market-values differs from valuing all shares at the H-share price. Both conventions need explicit labeling.',
                        'No intrayear share events, corporate-action completeness or historical quote vintage is certified here.'],
                prices_filled=False,cap_or_pool_computed=False,strategy_trials=0,observers_modified=False)
    with (OUT/'AUDIT.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],rows=rows,audit_sha256=sha(OUT/'AUDIT.json')),ensure_ascii=False))

if __name__=='__main__':main()
