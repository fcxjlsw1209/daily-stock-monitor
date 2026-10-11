"""Exchange calendar plus evidenced closures missing from pinned 4.5.6.

Only the union research uses this adapter; original paper accounts untouched.
"""
from pathlib import Path
import hashlib,json
import pandas as pd
import exchange_calendars as xc
P=Path(__file__).resolve().parent

def sessions(start,end):
    start=pd.Timestamp(start).normalize();end=pd.Timestamp(end).normalize()
    if start>end:raise ValueError('Reversed calendar range')
    calendar=xc.get_calendar('XNYS',start=start-pd.Timedelta(days=7),end=end+pd.Timedelta(days=7))
    result=calendar.sessions
    closed=[]
    for row in json.loads((P/'CALENDAR_EXCEPTIONS.json').read_text())['closures']:
        if hashlib.sha256((P/row['source_file']).read_bytes()).hexdigest()!=row['source_sha256']:
            raise ValueError('Calendar source checksum mismatch')
        closed.append(row['date'])
    return result[(result>=start)&(result<=end)&~result.strftime('%Y-%m-%d').isin(closed)]
