"""PepsiCo named consolidated statements; no expected amounts or candidate dates."""
import calendar
from datetime import date,datetime,timedelta
import re
from bs4 import BeautifulSoup
class ParseError(ValueError):pass
def require(ok,reason):
    if not ok:raise ParseError(reason)
def clean(x):return re.sub(r'\s+',' ',x).strip()
def last_saturday(year):
    x=date(year,12,31);return x-timedelta(days=(x.weekday()-5)%7)
def textual_date(month,day,year):
    month=month.rstrip('.');month='Sep' if month=='Sept' else month
    matches=[i for i in range(1,13) if month in (calendar.month_name[i],calendar.month_abbr[i])]
    require(len(matches)==1,'unknown_month')
    try:return date(int(year),matches[0],int(day))
    except ValueError as e:raise ParseError('invalid_date') from e
def identity(text):
    s=clean(text);start=s.find('PepsiCo Reports ');require(start>=0,'issuer_headline_missing')
    s=s[start:];stop=s.find('PepsiCo, Inc. and Subsidiaries');s=s[:stop] if stop>=0 else s
    matches=list(re.finditer(r'PURCHASE, N\.Y\.\s*(?:-|,)\s*([A-Za-z]+\.?)\s+(\d{1,2}),\s*(20\d{2})',s))
    require(len(matches)==1,'dated_issuer_not_unique');m=matches[0]
    nearby=s[m.end():m.end()+300]
    ids=list(re.finditer(r'PepsiCo, Inc\. \((NASDAQ|NYSE): PEP\) today reported',nearby))
    require(len(ids)==1,'dated_security_identity_missing')
    return dict(issuer='PepsiCo, Inc.',symbol='PEP',exchange=ids[0][1],release_date=textual_date(*m.groups()).isoformat(),dateline=m.group(),security_excerpt=ids[0].group(),continuous_identity_verified=False)
def statement(text,locator):
    s=clean(text)
    pat=r'PepsiCo, Inc\. and Subsidiaries\s+((?:Condensed )?Consolidated Statement of Income)\s+(.*?)\s+Cost of sales\b'
    matches=list(re.finditer(pat,s));require(len(matches)==1,'named_consolidated_statement_not_unique');m=matches[0]
    header,sep,row=m[2].partition('Net Revenue ');require(sep and 'Net Revenue ' not in row,'net_revenue_row_not_unique')
    units=re.match(r'\(in millions except per share amounts(?:, unaudited)?\)\s*(?:\(Unaudited\)\s*)?',header)
    require(units is not None and 'unaudited' in units.group().lower(),'printed_units_missing')
    h=header[units.end():].strip();datepat=r'\d{1,2}/\d{1,2}/20\d{2}'
    change_columns=h.count('Change')
    require(change_columns in (0,2),'change_column_count')
    if change_columns:require(re.fullmatch(r'12 Weeks Ended 36 Weeks Ended '+datepat+r' '+datepat+r' Change '+datepat+r' '+datepat+r' Change',h) is not None,'change_header_position')
    h=h.replace(' Change','');labels=clean(re.sub(datepat,'',h));dates=re.findall(datepat,h)
    require(len(dates) in (2,4) and clean(re.sub(datepat,'',h))==labels,'invalid_period_header')
    try:ends=[datetime.strptime(v,'%m/%d/%Y').date() for v in dates]
    except ValueError as e:raise ParseError('invalid_period_date') from e
    require(all(x.weekday()==5 for x in ends) and 350<=(ends[0]-ends[1]).days<=378,'period_direction_or_weekday')
    if len(ends)==4:require(ends[2:]==ends[:2],'quarter_ytd_end_disagreement')
    if labels=='Quarter Ended Year Ended':nominals=[None,52];require(len(ends)==4,'annual_column_count')
    else:
        labels=clean(labels);weeks=[int(x) for x in re.findall(r'(\d+) Weeks Ended',labels)]
        require(weeks in ([12],[12,24],[12,36]) and labels==' '.join(str(x)+' Weeks Ended' for x in weeks),'unsupported_week_labels');nominals=weeks
    number=r'(?:\d{1,3}(?:,\d{3})+|\d+)';tokens=re.findall(r'\$\s*('+number+r')',row)
    if change_columns:
        amt=r'\$\s*'+number;percent=r'(?:\(\d+(?:\.\d+)?\)|-?\d+(?:\.\d+)?)%'
        require(re.fullmatch(amt+r' '+amt+r' '+percent+r' '+amt+r' '+amt+r' '+percent,row.strip()) is not None,'change_value_position')
    else:require(re.sub(r'\$\s*'+number,'',row).strip()=='','net_revenue_value_columns')
    require(len(tokens)==len(ends),'net_revenue_value_columns')
    periods=[]
    for i,nominal in enumerate(nominals):
        annual=nominal==52;quarter_only=nominal is None
        period=dict(label='Year Ended' if annual else ('Quarter Ended' if quarter_only else str(nominal)+' Weeks Ended'),weeks_printed=None if annual or quarter_only else nominal,annual=annual,quarter_only=quarter_only,locator=locator,header_excerpt=header.strip(),row_excerpt='Net Revenue '+row.strip())
        for k,j in (('current',2*i),('prior',2*i+1)):
            end=ends[j];start=end-timedelta(days=nominal*7-1) if nominal and not annual else None
            period[k]=dict(end=end.isoformat(),start=start.isoformat() if start else None,revenue_usd=int(tokens[j].replace(',',''))*1000000,duration_days=nominal*7 if nominal and not annual else None,start_basis='printed weeks subtracted inclusively from printed end' if start else None)
        periods.append(period)
    return dict(heading=m[1],concept='consolidated_net_revenue',unit='USD',scale=1000000,currency_basis='printed dollar symbols on US issuer consolidated statement',periods=periods)
def parse_pdf(pages):
    require(type(pages)is list and pages and all(type(p)is str for p in pages),'invalid_pages')
    ident=identity(' '.join(pages));found=[]
    for i,p in enumerate(pages,1):
        if re.search(r'PepsiCo, Inc\. and Subsidiaries\s+(?:Condensed )?Consolidated Statement of Income',clean(p)):found.append(statement(p,dict(page_1based=i)))
    require(len(found)==1,'named_consolidated_statement_not_unique')
    return dict(identity=ident,statement=found[0])
def html_text(raw):
    soup=BeautifulSoup(raw,'html.parser')
    for x in list(soup.find_all(['script','style','noscript','template'])):x.decompose()
    return soup.get_text(' ',strip=True)
def parse_html(raw):
    text=html_text(raw);return dict(identity=identity(text),statement=statement(text,dict(format='visible_original_html')))
def parse_fiscal(pages):
    # Selected original pages: cover, calendar paragraph and signed signature.
    require(len(pages)==3,'fiscal_page_count');cover,body,signed=map(clean,pages)
    require('FORM 10-K' in cover and 'Commission file number 1-1183' in cover and 'PepsiCo, Inc.' in cover and '13-1584302' in cover,'fiscal_cover_identity')
    m=re.search(r'For the fiscal year ended ([A-Za-z]+) (\d{1,2}), (20\d{2})',cover);require(m is not None,'fiscal_cover_period');end=textual_date(*m.groups())
    require(end==last_saturday(end.year),'fiscal_cover_not_last_saturday')
    rule='Our fiscal year ends on the last Saturday of each December'
    require(rule in body,'fiscal_rule_missing')
    require('SIGNATURES' in signed and 'PepsiCo has duly caused this report' in signed,'fiscal_signature_missing')
    m=re.search(r'Dated: ([A-Za-z]+) (\d{1,2}), (20\d{2})',signed);require(m is not None,'fiscal_signature_date');stamp=textual_date(*m.groups())
    return dict(issuer='PepsiCo, Inc.',report_date=end.isoformat(),signature_date=stamp.isoformat(),rule=rule,rule_excerpt=body,period_start_directly_printed=False)
def with_calendar(parsed,calendar_evidence,filing_date):
    import copy
    out=copy.deepcopy(parsed)
    for p in out['statement']['periods']:
        if not p['annual']:continue
        require(calendar_evidence is not None,'annual_calendar_missing')
        require(calendar_evidence['report_date']==p['current']['end'] and p['current']['end']<calendar_evidence['signature_date']<=filing_date,'annual_calendar_not_causal_or_matching')
        for role in ('current','prior'):
            item=p[role];end=date.fromisoformat(item['end']);require(end==last_saturday(end.year),'annual_endpoint_rule_mismatch')
            start=last_saturday(end.year-1)+timedelta(days=1);days=(end-start).days+1
            require(days in (364,371),'invalid_annual_duration')
            item.update(start=start.isoformat(),duration_days=days,start_basis='day after prior December last Saturday under selected original10K fiscal rule',fiscal_evidence=calendar_evidence)
    return out
