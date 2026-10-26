"""Small candidate set extrapolated from already observed official CDN names.
Every successful response must be an actual PDF and is validated from content.
A name/date in a URL is never itself event evidence; 403/429 stops the host.
"""
import collect as c
urls=[
'/files/doc_financials/2016/q1/1Q2016-10-Q.pdf',
'/files/doc_financials/2021/q1/10Q-1Q2021.pdf',
'/files/doc_news/Johnson--Johnson-Reports-Q1-2025-Results-2025.pdf',
'/files/doc_news/Johnson--Johnson-Reports-Q2-2025-Results-Raises-2025-Outlook-2025.pdf',
'/files/doc_news/Johnson--Johnson-reports-Q2-2026-results-raises-2026-outlook-2026.pdf',
'/files/doc_news/Johnson--Johnson-reports-Q2-2024-results-2024.pdf',
'/files/doc_news/Johnson--Johnson-Reports-Q2-2023-Results-2023.pdf',
'/files/doc_news/Johnson--Johnson-Reports-Q3-2023-Results-2023.pdf',
'/files/doc_financials/2024/q1/10-Q.pdf',
'/files/doc_financials/2026/q2/2026-Q2-Form-10-Q-28Jun2026-07232026.pdf',
'/files/doc_financials/2012/q1/10-Q.pdf',
'/files/doc_financials/2012/q2/10-Q.pdf',
'/files/doc_financials/2012/q3/10-Q.pdf',
]
for suffix in urls:
 u=c.CDN+suffix
 try:
  p,m=c.fetch(u,'candidate_'+c.sha(u.encode())[:16]);print(u,len(p.read_bytes()),c.pdf_text(p)[:120].replace('\n',' '),flush=True)
 except Exception as e:
  print('ERROR',u,str(e),flush=True)
  if getattr(e,'code',None) in (403,429) or isinstance(e,RuntimeError) and 'source stopped' in str(e):break
