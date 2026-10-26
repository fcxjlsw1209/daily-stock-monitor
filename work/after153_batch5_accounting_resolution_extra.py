"""Only earlier pinned-local release context; no amount extraction."""
import hashlib,json,re
from bs4 import BeautifulSoup
import after153_batch5_accounting_resolution as m
EXTRA=[('TMO','c1163eb22b3373bd49f9'),('PYPL','c5dd6b4c3afd11106b43')]
rows=[];pins={}
for issuer,stem in EXTRA:
    name=f'outputs/monthly_earnings_repair/batch_financial_consumer/{issuer}/raw/{stem}.html'
    receipt=json.loads(m.read(name+'.meta.json',pins));raw=m.read(name,pins,receipt['sha256'])
    m.check(receipt['status']==200,'receipt status');text=raw.decode();parser=m.Blocks(text);parser.feed(text)
    passages=[]
    for start,end,tag in sorted(parser.blocks):
        visible=m.norm(BeautifulSoup(text[start:end],'html.parser').get_text(' ',strip=True))
        if not re.search(r'basis of presentation|combined|prior to.{0,50}separation|service cost|pension|2017.07|capitaliz|historical|allocation',visible,re.I) or len(visible)>6000:continue
        a=len(text[:start].encode());b=len(text[:end].encode())
        passages.append(dict(path=name,sha256=receipt['sha256'],locator=dict(kind='raw_utf8_byte_span',start=a,end_exclusive=b,tag=tag),raw_span_sha256=hashlib.sha256(raw[a:b]).hexdigest(),quote_numbers_redacted=m.redact(visible)))
    rows.append(dict(issuer=issuer,url=receipt['url'],title=BeautifulSoup(text,'html.parser').title.get_text(),passages=passages))
print(json.dumps(dict(reports=rows,input_hashes=pins),ensure_ascii=False,indent=2))
