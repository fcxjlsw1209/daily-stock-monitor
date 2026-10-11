"""Prioritize quote gaps without erasing halted securities or marking admission.

A terminal quarantine is not proof that a security was untradeable all that day.
Only an explicit last-trading date proves subsequent dates lack a market quote.
Cash/stock receivables remain required even on verified non-trading dates.
"""
from pathlib import Path
from collections import Counter, defaultdict
import datetime, hashlib, json
P=Path(__file__).resolve().parent

def classify(day,event):
    if not event:return 'unresolved_quote_or_identity'
    last=event.get('last_trading_date')
    if last and day>last:return 'verified_post_trading_action_valuation_required'
    if day>=event['quarantine_from']:return 'terminal_timing_or_valuation_unresolved'
    return 'unresolved_quote_or_identity'

def main():
    inputs=['ALIAS_COVERAGE_DIAGNOSTIC.json','REVIEWED_TERMINAL_EVENTS.json']
    events={r['symbol']:r for r in json.loads((P/inputs[1]).read_text())}
    for e in events.values():
        if hashlib.sha256((P/e['source_file']).read_bytes()).hexdigest()!=e['source_sha256']:
            raise ValueError('Terminal source checksum mismatch: '+e['symbol'])
    rows=defaultdict(lambda:dict(indexes=set(),categories=set()))
    for interval in json.loads((P/inputs[0]).read_text())['intervals']:
        for day in interval['missing_dates_after_alias_candidates']:
            symbol=interval['symbol'];r=rows[(symbol,day)]
            r['indexes'].add(interval['index']);r['categories'].add(classify(day,events.get(symbol)))
    result=[]
    for (symbol,day),r in sorted(rows.items()):
        if len(r['categories'])!=1:raise ValueError('Conflicting lifecycle categories')
        result.append(dict(symbol=symbol,date=day,indexes=sorted(r['indexes']),category=next(iter(r['categories']))))
    counts=Counter(r['category'] for r in result)
    symbols=Counter(r['symbol'] for r in result)
    report=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        admission=False,unique_symbol_session_gaps=len(result),category_counts=dict(counts),
        symbol_counts=dict(symbols.most_common()),gaps=result,
        source_sha256={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in inputs},
        limitations=['Potential alias matches are not identity-certified price coverage.',
        'Verified post-trading dates still require held-position settlement; none are waived.',
        'Warmup, missing member events, units, industry and unidentified sources are outside this diagnostic.'])
    (P/'LIFECYCLE_GAPS_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['gaps','source_sha256']},indent=2))
if __name__=='__main__':main()
