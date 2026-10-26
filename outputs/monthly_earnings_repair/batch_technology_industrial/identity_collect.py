from join_collect import *
def skhy():
 p=HERE/'SKHY/raw/q226_actual_release.bin';m=json.loads(p.with_suffix('.meta.json').read_text());t=clean(extract(p))
 d=re.search(r'Seoul, July 29, 2026',t);assert d and '2Q26 Financial Results' in t
 sp=HERE/'SKHY/raw/q226_period_notice.bin';sm=json.loads(sp.with_suffix('.meta.json').read_text());st=text(sp.read_bytes())
 match=re.search('截止2026年6月30日的2026年第二季度财务报告',st);assert match
 r=dict(release_date='2026-07-29',fiscal_period_end_actual='2026-06-30',fiscal_year=2026,fiscal_quarter=2,dateline_excerpt=d.group(),period_excerpt=match.group(),identity_excerpt='SK hynix Announces 2Q26 Financial Results')
 e=event('SKHY',p,m,r);proof=dict(evidence_path=rel(sp),sha256=sm['sha256'],source_url=sm['url'],excerpt=match.group())
 e['fiscal_period_evidence']=dict(proof,basis='Issuer original notice explicitly identifies the already-ended June 30 reporting period; subsequently released actual results separately verify publication. No scheduled clock or publication date substituted.')
 identity=HERE/'SKHY/raw/adr_identity.bin';im=json.loads(identity.with_suffix('.meta.json').read_text())
 e['additional_evidence']=[proof,dict(evidence_path=rel(identity),sha256=im['sha256'],source_url=im['url'],excerpt='Nasdaq ADR listing began July 10, 2026')]
 e['security_identity']={'issuer':'SK hynix Inc.','instrument':'Nasdaq ADR','listing_date':'2026-07-10','historical_ticker_warning':'Do not treat pre-ADR Korean ordinary-share history as a historical US-listed SKHY security.','frozen_candidate_window':['2026-07-31','2026-08-31']}
 merge('SKHY',[e],[])
if __name__=='__main__':skhy()
