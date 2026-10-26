from pathlib import Path
P=Path('outputs/monthly_sec_visa_cdn_admission_v1');Q=Path('outputs/monthly_sec_pfizer_cdn_admission_v1')
s=(Q/'test_admission.py').read_text().replace('pfe_test_','visa_test_').replace("symbol='PFE'","symbol='V'").replace("cik='0000078003'","cik='0001403161'")
s=s.replace("default=payload();parsed=a.parser.parse_pdf(default,filed);period=parsed['statement']['periods'][0]","default=payload(2,2018);parsed=a.parser.parse_pdf(default,filed);period=parsed['statement']['periods'][-1]")
s=s.replace('s206.q4cdn.com/795948973','s1.q4cdn.com/050606653').replace('795948973','050606653')
a=s.index(" nsraw=b'%PDF-namespace'");b=s.index('\n design=dict',a)
s=s[:a]+''' nsraw=b"<html><head><title>Visa Expands Real-Time Funds Disbursement Offering in Canada with Moneris | Visa</title></head><body><a href='https://s1.q4cdn.com/050606653/files/doc_financials/2019/q4/CORRECTED-TRANSCRIPT-Visa,-Inc.(V-US),-Q4-2019-Earnings-Call,-24-October-2019-5-00-PM-ET.pdf'>Transcript</a><p>Data refers to Visa&#8217;s Fiscal Year 2019, which ended September 2019.</p></body></html>";docs['ns/raw/main.decoded']=docs['ns/raw/main.response']=nsraw;nsurl='https://www.visa.ca/fixture.html';add('ns/raw/main.meta.json',dict(url=nsurl,final_url=nsurl,status=200,decoded_sha256=a.sha(nsraw),response_sha256=a.sha(nsraw),decoded_bytes=len(nsraw),response_bytes=len(nsraw),decoder='identity',content_encoding=None,content_type='text/html'));hashes={n:a.sha(v)for n,v in docs.items()}''' +s[b:]
a=s.index(' with patch.object(b,');b=s.index('\nclass AdmissionTests',a)
s=s[:a]+''' with patch.object(b,'module',return_value=a):
  return{n:a.strict_json(x)for n,x in b.produce(d,docs,fake_pdf).items()}
''' +s[b:]
a=s.index(' def test_wrong_amount_rejected(');b=s.index(' def test_wrong_cdn_tenant(',a)
s=s[:a]+''' def test_wrong_amount_rejected(self):
  x=payload(2,2018);x['pages'][1]['text']=x['pages'][1]['text'].replace('$ 4,400','$ 4,401');d,docs=synthetic(x);self.assertEqual(outputs(d,docs)['COVERAGE.json']['unverified_count'],1)
 def test_late_version_unverified(self):
  x=payload(2,2018);x['metadata']['ModDate']='D:20180810120000Z';d,docs=synthetic(x);self.assertEqual(outputs(d,docs)['COVERAGE.json']['unverified_count'],1)
 def test_missing_version_not_admitted(self):
  x=payload(2,2018);x['metadata']={};d,docs=synthetic(x);self.assertEqual(outputs(d,docs)['COVERAGE.json']['unverified_count'],1)
''' +s[b:]
(P/'test_admission.py').write_text(s)
