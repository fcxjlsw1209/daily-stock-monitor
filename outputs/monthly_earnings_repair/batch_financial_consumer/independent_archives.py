import sys,json,re,urllib.parse
import core as c
from archive_download import links
from expand_download import saveq
URLS={'PEP':'https://investor.pepsico.com/investors/earnings/index.html','ABT':'https://abbott.mediaroom.com/press-releases?category=783&l=100','DIS':'https://investors.thewaltdisneycompany.com/financials/quarterly-results/default.aspx','BAC':'https://newsroom.bankofamerica.com/category/press-release-category/corporate-and-financial-news','CMCSA':'https://corporate.comcast.com/news-information/financial-statements','BUD':'https://www.ab-inbev.com/investors/results-center'}
for s,u in URLS.items():
 try:
  ls=links(u,s,'independent_archive');c.dump(c.HERE/s/'independent_links.json',ls);print(s,len(ls),flush=True)
  if s=='PEP':selected=[a for a in ls if re.search('Press Release',a['text'],re.I)]
  elif s=='ABT':selected=[a for a in ls if re.search(r'Abbott Reports .*Quarter',a['text'],re.I)]
  elif s=='DIS':selected=[a for a in ls if re.search('earnings.release|earnings.report|press.release',a['url']+' '+a['text'],re.I)]
  elif s=='BAC':selected=[a for a in ls if re.search(r'Bank of America Reports .*Quarter.*Financial Results',a['text'],re.I)]
  elif s=='CMCSA':selected=[a for a in ls if re.search('release|result',a['url']+' '+a['text'],re.I)]
  else:selected=[a for a in ls if '.pdf' in a['url'] and re.search('result|release',a['text'],re.I)]
  saveq(s,selected)
 except Exception as e:print(s,'FAILED',str(e),flush=True)
# HSBC public pagination is linked in archive itself.
s='HSBC';ls=[]
for page in [1,2]:
 try:ls+=links(f'https://www.hsbc.com/investors/results-and-announcements/all-reporting/group?page={page}&take=100',s,'page_'+str(page))
 except Exception as e:print(s,str(e),flush=True)
selected=[a for a in ls if a['url'].endswith('.pdf') and re.search(r'(Earnings Release|media release)',a['text']) and not re.search(r'Chinese|Form 6-K|Transcript',a['text'],re.I)]
saveq(s,selected)
