"""Pure Apple publication, statement and fiscal-boundary parser. No network or I/O data reads."""
from datetime import date, datetime
import importlib.util
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
FROZEN_PARSER='outputs/monthly_sec_apple_ir_crosscheck_v2/parser.py'
_spec=importlib.util.spec_from_file_location('frozen_apple_statement',ROOT/FROZEN_PARSER)
statement=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(statement)
ParseError=ValueError
MONTHS='January|February|March|April|May|June|July|August|September|October|November|December'
HEADLINE=r'Apple Reports (?:Record )?(First|Second|Third|Fourth) Quarter Results'

def require(ok,reason):
    if not ok:raise ParseError(reason)

def norm(text):return ' '.join(text.split())

def publication(raw,url):
    soup=BeautifulSoup(raw,'html.parser');heads=soup.find_all('h1');visible=soup.select('.category-eyebrow__date')
    require(len(heads)==len(visible)==1,'publication_or_headline_not_unique')
    headline=norm(heads[0].get_text(' ',strip=True));match=re.fullmatch(HEADLINE,headline,re.I)
    require(match is not None,'issuer_results_headline_missing')
    printed=norm(visible[0].get_text(' ',strip=True));day=datetime.strptime(printed,'%B %d, %Y').date().isoformat()
    articles=[]
    for s in soup.find_all('script',type='application/ld+json'):
        value=json.loads(s.string or s.get_text());values=value if isinstance(value,list) else [value]
        articles.extend(v for v in values if isinstance(v,dict) and v.get('@type')=='NewsArticle')
    require(len(articles)==1,'news_article_not_unique');article=articles[0]
    require(article.get('datePublished')==day+'Z' and article.get('mainEntityOfPage')==url and norm(article.get('headline',''))==headline,'publication_metadata_mismatch')
    require(article.get('author')=={'@id':'https://www.apple.com/#organization'} and article.get('publisher')=={'@id':'https://www.apple.com/#organization'},'publisher_mismatch')
    for item in soup(['script','style','nav','footer']):item.decompose()
    text=norm(soup.get_text(' ',strip=True));ends=[]
    patterns=[r'fiscal year ended\s+('+MONTHS+r')\s+(\d{1,2}),?\s+(\d{4})',r'fiscal\s+\d{4}\s+fourth quarter ended\s+('+MONTHS+r')\s+(\d{1,2}),?\s+(\d{4})']
    for pattern in patterns:
        for m in re.finditer(pattern,text,re.I):
            end=datetime.strptime(' '.join(m.groups()),'%B %d %Y').date()
            require(end.weekday()==5 and end.month in (9,10) and end.isoformat()<day,'invalid_explicit_fiscal_end')
            item=dict(fiscal_year=end.year,end=end.isoformat(),basis='explicit_original_release_text',excerpt=m.group(),directly_printed=True)
            if item not in ends:ends.append(item)
    return dict(issuer='Apple Inc.',headline=headline,release_date=day,fiscal_quarter=['first','second','third','fourth'].index(match[1].lower())+1,publication_basis='visible_date_and_NewsArticle_datePublished',historical_web_snapshot_verified=False,exact_publication_timestamp_verified=False,year_ends=ends)

def financial(payload):
    result=statement.parse_statement(payload['text'],page1_chars=payload['chars'],page_bbox=payload['bbox'])
    ends=[]
    if max(result['duration_months'])==12:
        for role in ('current','prior'):
            end=date.fromisoformat(result[role+'_end']);require(end.weekday()==5 and end.month in (9,10),'invalid_annual_end')
            ends.append(dict(fiscal_year=end.year,end=end.isoformat(),basis='explicit_twelve_month_header',excerpt=result['header_durations'],directly_printed=True))
    return dict(statement=result,year_ends=ends)
