"""Public S&P Global history via Stock Analysis; no credentials or access bypass."""
import hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

FIELDS=['Open','High','Low','Close','Adj Close','Volume']

def parse_history(html,cutoff):
 soup=BeautifulSoup(html,'html.parser');rows=[]
 for table in soup.find_all('table'):
  headers=[x.get_text(' ',strip=True) for x in table.find_all('th')]
  if headers!=['Date','Open','High','Low','Close','Adj. Close','Change','Volume']:continue
  for tr in table.find_all('tr'):
   cells=[x.get_text(' ',strip=True) for x in tr.find_all('td')]
   if len(cells)!=8:continue
   day=pd.to_datetime(cells[0],format='%b %d, %Y')
   if day>pd.Timestamp(cutoff):continue
   values=[float(cells[i].replace(',','').replace('$','')) for i in [1,2,3,4,5,7]]
   if not np.isfinite(values).all() or min(values[:5])<=0 or values[5]<0:raise ValueError('Invalid secondary quote')
   o,h,l,c,adj,vol=values
   if not l<=min(o,c)<=max(o,c)<=h:raise ValueError('Secondary OHLC inconsistency')
   rows.append(dict(Date=day,**dict(zip(FIELDS,values))))
 if not rows:raise ValueError('Secondary history table missing or incomplete')
 frame=pd.DataFrame(rows).set_index('Date').sort_index()
 if frame.index.has_duplicates:raise ValueError('Duplicate secondary dates')
 return frame

def fetch_page(url,cache):
 cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
 response=requests.get(url,timeout=20,impersonate='chrome');response.raise_for_status()
 body=response.text;digest=hashlib.sha256(body.encode()).hexdigest()
 (cache/(digest+'.html')).write_text(body)
 return body,{'url':url,'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(),'sha256':digest,'raw_file':str(cache/(digest+'.html'))}

def verify_no_actions(dividend_html,statistics_html,start,end):
 soup=BeautifulSoup(dividend_html,'html.parser');text=soup.get_text(' ',strip=True)
 if 'There is no dividend history available' not in text:
  tables=[t for t in soup.find_all('table') if 'Cash Amount' in t.get_text() and 'Record Date' in t.get_text()]
  if len(tables)!=1:raise ValueError('Cannot establish dividend history')
  dates=[]
  for tr in tables[0].find_all('tr'):
   cells=tr.find_all('td')
   if cells:dates.append(pd.to_datetime(cells[0].get_text(' ',strip=True),format='%b %d, %Y'))
  if not dates or min(dates)>start:raise ValueError('Dividend history does not cover gap')
  if any(start<=d<=end for d in dates):raise ValueError('Dividend in recovery window; manual adjustment required')
 text=BeautifulSoup(statistics_html,'html.parser').get_text(' ',strip=True)
 match=re.search(r'Last Split Date ([A-Z][a-z]{2} \d{1,2}, \d{4})',text)
 if match:
  if pd.to_datetime(match.group(1),format='%b %d, %Y')>=start:raise ValueError('Split in/after recovery window; manual adjustment required')
 elif not re.search(r'(has never split|has not split|no stock splits)',text,re.I):
  raise ValueError('Cannot establish split history')

def recover(frame,symbol,missing,cutoff,cache):
 # Only explicitly verified stock pages. No guessed ETF corporate-action metadata.
 slug=symbol.lower().replace('-','.')
 base='https://stockanalysis.com/stocks/'+slug+'/'
 history,p1=fetch_page(base+'history/',cache)
 soup=BeautifulSoup(history,'html.parser');text=soup.get_text(' ',strip=True)
 if not re.search(r'\('+re.escape(symbol.replace('-','.'))+r'\)',text,re.I) or 'USD' not in text:
  raise ValueError('Secondary symbol/currency mismatch')
 other=parse_history(history,cutoff)
 if not missing.isin(other.index).all():raise ValueError('Secondary source lacks missing sessions')
 anchors=frame.index.intersection(other.index)
 anchors=anchors[anchors<missing.min()][-5:]
 if len(anchors)<5 or not np.allclose(frame.loc[anchors,FIELDS[:5]],other.loc[anchors,FIELDS[:5]],rtol=1e-6,atol=.011):
  raise ValueError('Secondary price/adjustment basis mismatch')
 dividend,p2=fetch_page(base+'dividend/',cache);stats,p3=fetch_page(base+'statistics/',cache)
 verify_no_actions(dividend,stats,missing.min(),pd.Timestamp(cutoff))
 rows=other.loc[missing].copy();rows['Dividends']=0.;rows['Stock Splits']=0.
 if 'Capital Gains' in frame:raise ValueError('Fund distribution metadata unavailable from stock fallback')
 result=pd.concat([frame,rows]).sort_index()
 return result,{'symbol':symbol,'sessions':[str(d.date()) for d in missing],'provider':'Stock Analysis / S&P Global','sources':[p1,p2,p3],'note':'Published daily Open; not certified Nasdaq official opening auction price'}
