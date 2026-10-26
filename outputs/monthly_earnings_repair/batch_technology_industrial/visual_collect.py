"""Exact original-PDF page transcriptions verified visually after broken embedded fonts."""
from join_collect import *
def netflix_2011q3():
 p=HERE/'NFLX/raw/08abf8d0d66428a108ca.bin';m=json.load(open(p.with_suffix('.meta.json')))
 original_sha=sha(p.read_bytes())
 proofpath=HERE/'NFLX/visual_verification_2011q3.json'
 if not proofpath.exists():
  images=[p.with_suffix('.page1.png'),p.with_suffix('.page13.png')];ocr=p.with_suffix('.ocr.txt')
  dump(proofpath,dict(source_url=m['url'],source_sha256=original_sha,verification='Both original-PDF rendered pages inspected visually: page 1 dated shareholder letter; page 13 consolidated statement first current Three Months Ended column.',renderer='PDFKit thumbnail; Apple Vision accurate en-US OCR, no autocorrection',ocr_path=rel(ocr),ocr_sha256=sha(ocr.read_bytes()),rendered_pages=[dict(evidence_path=rel(x),sha256=sha(x.read_bytes())) for x in images],release_date='2011-10-24',release_excerpt='Page 1: October 24th, 2011; Dear Fellow Shareholders',period_end='2011-09-30',period_excerpt='Page 13: Netflix, Inc. Consolidated Statements of Operations (unaudited); Three Months Ended; first column September 30, 2011',verified_at=now()))
 v=json.load(open(proofpath));assert v['source_sha256']==original_sha
 assert sha((ROOT/v['ocr_path']).read_bytes())==v['ocr_sha256']
 for im in v['rendered_pages']:assert sha((ROOT/im['evidence_path']).read_bytes())==im['sha256']
 r=dict(release_date=v['release_date'],fiscal_period_end_actual=v['period_end'],fiscal_year=2011,fiscal_quarter=3,dateline_excerpt=v['release_excerpt'],period_excerpt=v['period_excerpt'],identity_excerpt='Original quarterly financial statement current column September 30, 2011: calendar fiscal FY2011 Q3')
 e=event('NFLX',p,m,r,'issuer_original_shareholder_letter_full_bytes_visually_verified')
 e['release_date_basis']='Date printed on original shareholder-letter page 1, checked visually because embedded-font extraction duplicated characters.'
 e['additional_evidence']=[dict(evidence_path=rel(proofpath),sha256=sha(proofpath.read_bytes()),source_url=m['url'],excerpt='Visual verification of original pages 1 and 13')]+v['rendered_pages']
 merge('NFLX',[e],[])
if __name__=='__main__':netflix_2011q3()

def ibm_2015q3():
 p=HERE/'IBM/raw/4f1c38b1993d1a8c8f78.bin';m=json.load(open(p.with_suffix('.meta.json')))
 vpath=HERE/'IBM/visual_verification_2015q3.json';ocr=p.with_suffix('.ocr.txt');images=[p.with_suffix('.page1.png'),p.with_suffix('.page7.png')]
 if not vpath.exists():dump(vpath,dict(source_url=m['url'],source_sha256=sha(p.read_bytes()),verification='Original PDF rendered pages visually inspected; OCR ordering is not used to choose a financial table column.',renderer='PDFKit + Apple Vision accurate en-US, no autocorrection',ocr_path=rel(ocr),ocr_sha256=sha(ocr.read_bytes()),rendered_pages=[dict(evidence_path=rel(x),sha256=sha(x.read_bytes())) for x in images],release_date='2015-10-19',release_excerpt='Page 1: ARMONK, N.Y., October 19, 2015 ... IBM (NYSE: IBM) today announced third-quarter 2015 diluted earnings',period_end='2015-09-30',period_excerpt='Page 7: Comparative Financial Results, Three Months Ended September 30; first current-year column 2015; separate Nine Months Ended columns on the right',verified_at=now()))
 v=json.load(open(vpath));assert v['source_sha256']==sha(p.read_bytes()) and v['ocr_sha256']==sha(ocr.read_bytes())
 for x in v['rendered_pages']:assert sha((ROOT/x['evidence_path']).read_bytes())==x['sha256']
 r=dict(release_date=v['release_date'],fiscal_period_end_actual=v['period_end'],fiscal_year=2015,fiscal_quarter=3,dateline_excerpt=v['release_excerpt'],period_excerpt=v['period_excerpt'],identity_excerpt='Page 1 headline: IBM REPORTS 2015 THIRD-QUARTER RESULTS')
 e=event('IBM',p,m,r,'issuer_original_release_pdf_full_bytes_visually_verified');e['coverage_scope']='training_candidate_history_only';e['release_date_basis']='Original dated results-release page 1, visually verified from the image-only original PDF.'
 e['additional_evidence']=[dict(evidence_path=rel(vpath),sha256=sha(vpath.read_bytes()),source_url=m['url'],excerpt='Original pages 1 and 7 visually checked')]+v['rendered_pages'];merge('IBM',[e],[])
