import collect as c,build as b
u='https://data.alibabagroup.com/data/document/1490841477796855808.json'
p,m=c.fetch(u,'BABA','Original issuer IPO announcement API');tx=b.text(p,'BABA')
es=[e for e in c.load(c.HERE/'events.json',[]) if e['ticker']=='BABA'];first=min(es,key=lambda e:e['release_date']);a=next(a for a in c.load(c.HERE/'BABA_candidates.json',[]) if 'q2014/' in a.get('archive_url',''))
out=dict(ticker='BABA',candidate_date='2014-10-31',classification='structural_before_first_verified_regular_listed_quarterly_release',scope='Regular listed-issuer earnings releases; IPO registration financials are a different publication class.',first_regular_release_date=first['release_date'],first_regular_fiscal_year=first['fiscal_year'],first_regular_fiscal_quarter=first['fiscal_quarter'],first_regular_period_end=first['fiscal_period_end_actual'])
out['ipo_evidence']=dict(source_url=u,evidence_path=c.rel(p),sha256=m['sha256'],retrieved_at=m['retrieved_at'],source_representation='original_public_api_json_bytes',excerpt=tx[:1000])
out['first_regular_release_evidence']=dict(source_url=first['source_url'],evidence_path=first['evidence_path'],sha256=first['evidence_sha256'],excerpt=first['release_date_evidence'][0]['excerpt'])
out['official_2014_archive_evidence']=dict(source_url=a['archive_url'],evidence_path=a['archive_path'],sha256=a['archive_sha256'])
out['interpretation']='The official 2014 quarterly-results archive starts with September-quarter results, issued November 4, 2014. The October 31 candidate occurs after the September IPO but before this regular release. Preserve the event as unavailable for that candidate; do not fill it with the later release or pretend that IPO registration financial information was absent.'
out['limitation']='Archive boundary supports the listed quarterly-release scope, not a universal proof that the issuer had never disclosed any financial information before November 4.'
c.dump(c.HERE/'BABA_FIRST_LISTED_EARNINGS.json',out);print(m['bytes'],tx[:800])
