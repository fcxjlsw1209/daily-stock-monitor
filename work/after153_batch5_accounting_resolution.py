"""Two-report source-context resolution only. No financial-cell parser."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import sys
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
PRIOR='work/after153_batch5_accounting_feasibility_review.json'
PRIOR_SHA='6e8cc03a6502b8194792b6d17e65f3c5fec993c7bd638ee44c4fee4a4bb08b94'
OUT='work/after153_batch5_accounting_resolution.json'
TARGETS=('PYPL_2016-01-27','TMO_2019-01-30')
PATTERN=re.compile(r'combined|separat|stand.alone|eBay|historical|pro.forma|accounting|pension|retrospect|restat|reclassif|capitaliz|GAAP|December',re.I)

def check(ok,msg):
    if not ok:raise ValueError(msg)

def read(name,pins,sha=None):
    p=ROOT/name
    check(not p.is_symlink() and p.resolve().is_relative_to(ROOT),'Local regular source')
    raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
    check(sha is None or sha==h,'SHA mismatch '+name)
    check(name not in pins or pins[name]==h,'Changed '+name)
    pins[name]=h;return raw

def redact(text):
    return re.sub(r'\d[\d,.]*','[number]',text)

def norm(text):return ' '.join(text.split())

class Blocks(HTMLParser):
    def __init__(self,text):
        super().__init__(convert_charrefs=False)
        self.text=text;self.starts=[0];self.stack=[];self.blocks=[]
        for m in re.finditer('\n',text):self.starts.append(m.end())
    def pos(self):
        line,col=self.getpos();return self.starts[line-1]+col
    def handle_starttag(self,tag,attrs):
        if tag in ('p','td','th','h1','h2','h3','h4','li'):self.stack.append((tag,self.pos()))
    def handle_endtag(self,tag):
        if tag not in ('p','td','th','h1','h2','h3','h4','li'):return
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0]==tag:
                _,start=self.stack.pop(i);end=self.text.find('>',self.pos())+1
                self.blocks.append((start,end,tag));break

def scan():
    pins={};old=json.loads(read(PRIOR,pins,PRIOR_SHA));out=[]
    for event in old['event_reviews']:
        if event['event_id'] not in TARGETS:continue
        s=event['source'];raw=read(s['path'],pins,s['sha256']);r=json.loads(read(s['receipt_path'],pins,s['receipt_sha256']))
        check(r['sha256']==s['sha256'] and r['url']==s['url'] and r.get('status',r.get('http_status'))==200,'Receipt mismatch')
        text=raw.decode('utf-8');blocks=Blocks(text);blocks.feed(text)
        passages=[];seen=set()
        for start,end,tag in sorted(blocks.blocks):
            fragment=text[start:end];visible=norm(BeautifulSoup(fragment,'html.parser').get_text(' ',strip=True))
            if not PATTERN.search(visible) or len(visible)>5000 or visible in seen:continue
            # Keep small textual blocks; financial rows are never interpreted.
            if tag in ('td','th') and re.fullmatch(r'[\d$%,.()\s–—-]+',visible):continue
            seen.add(visible);a=len(text[:start].encode('utf-8'));b=len(text[:end].encode('utf-8'))
            check(raw[a:b]==fragment.encode('utf-8'),'Byte map')
            passages.append(dict(path=s['path'],sha256=s['sha256'],locator=dict(kind='raw_utf8_byte_span',start=a,end_exclusive=b,tag=tag),raw_span_sha256=hashlib.sha256(raw[a:b]).hexdigest(),quote_numbers_redacted=redact(visible)))
        out.append(dict(event_id=event['event_id'],source=s,passages=passages))
    check(tuple(e['event_id'] for e in out)==TARGETS,'Fixed targets')
    return out,pins

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    reports,pins=scan()
    if args.publish:raise ValueError('Manual resolutions not yet written')
    print(json.dumps(reports,ensure_ascii=False,indent=2))
