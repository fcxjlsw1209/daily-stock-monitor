"""Import specifically reviewed single-replacement press-release excerpts.

This parser fails closed on unexpected syntax; it does not certify an entire
membership chain or assume that a ticker identifies the intended security.
"""
from pathlib import Path
import re
import json
import hashlib
from dateutil.parser import parse
from zoneinfo import ZoneInfo

P=Path(__file__).resolve().parent


def main():
    file=P/'ndx_primary_reviewed_rows.json'
    rows=json.loads(file.read_text())
    seen={(r['effective_session'],r['action'],r['ticker_as_announced_or_effective']) for r in rows}
    count=0
    for f in sorted((P/'sources').glob('ndx_single_batch*.txt')):
        text=f.read_text()
        if 'NEW YORK,' not in text: continue
        body=text[text.index('NEW YORK,'):]
        date=re.search(r'NEW YORK, (.+?) \(GLOBE NEWSWIRE\)',body).group(1)
        effective=re.search(r'prior to market open on \w+, (.+?\d{4})',body).group(1)
        before,after=body.split('will become a component',1)
        adds=re.findall(r'\(Nasdaq:\s*([A-Z]+)\)',before)
        rem=re.search(r'(?:will replace|replacing) .+?\(Nasdaq:\s*([A-Z]+)\)',after).group(1)
        add=adds[-1]
        eff=parse(effective).date().isoformat()
        pub=parse(date).replace(hour=23,minute=59,second=59,tzinfo=ZoneInfo('America/New_York')).isoformat()
        url=re.search(r'\((https://[^\s]+)\)',text).group(1)
        for action,t in [('addition',add),('deletion',rem)]:
            key=(eff,action,t)
            if key in seen: continue
            rows.append(dict(effective_session=eff,action=action,ticker_as_announced_or_effective=t,
                effective_convention='prior_to_open',published_at=pub,source_url=url,
                source_file=str(f.relative_to(P)),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                source_representation='web_tool_excerpt_receipt',announcement_content_reviewed=True,
                identity_verified=False,chain_verified=False))
            seen.add(key);count+=1
    file.write_text(json.dumps(rows,indent=2)+'\n')
    print('Added',count,'Total',len(rows))


if __name__=='__main__':main()
