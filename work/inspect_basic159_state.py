"""Bounded metadata inspection, no share or return calculation."""
import json
from pathlib import Path

r=Path(__file__).resolve().parents[1]
def read(p):return json.loads((r/p).read_text())
s=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
print('STATE',json.dumps({k:s.get(k) for k in ('current','current_status','registered_queue')},ensure_ascii=False))
print('COUNTS',len(s['completed']),sum(x['trials'] for x in s['completed']))
print('PREVIOUS',json.dumps(s['completed'][-1],ensure_ascii=False))
raw=read('outputs/after158_basic_shares_extraction_v1/RAW_SHARES.json')['events']
print('EVENT_METADATA',json.dumps([{k:e[k] for k in ('event_id','symbol','fiscal_year','periods','inherited_holds')} for e in raw[:2]],ensure_ascii=False))
print('HOLDS',json.dumps([(e['event_id'],e['inherited_holds'].get('inherited_rejection_codes')) for e in raw if e['inherited_holds'].get('inherited_rejection_codes')],ensure_ascii=False))
print('GENERIC_UNIT_HEADERS',json.dumps([(e['event_id'],e['cells']['current']['unit'],e['cells']['current']['semantic_basis']) for e in raw if e['cells']['current']['unit']['share_specific'] is False],ensure_ascii=False))
print('CELLS_SCHEMA',json.dumps({k:raw[1]['cells']['current'].get(k) for k in raw[1]['cells']['current'] if k not in ('displayed_value','value_decimal','value_in_shares')},ensure_ascii=False))
for f in ('work/after158_basic_shares_independent_review.json','work/after158_basic_shares_independent_remaining_review.json'):
 d=read(f);print(f,'keys',list(d));print('SUMMARY',json.dumps({k:v for k,v in d.items() if k in ('passed','summary','checks','status')},ensure_ascii=False))
