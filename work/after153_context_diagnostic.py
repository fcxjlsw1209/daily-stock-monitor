import json
import after153_next_source_context as m
r=m.build()
for e in r['events']:
 if not e['statement_contexts']:continue
 print(json.dumps(dict(id=e['event_id'],expected_end=e['official_actual_period_end'],tables=[dict(table=t['table'],kind=t['kind'],dates=t['date_columns'],years=t['year_columns'],annual=t['annual_header_locators'],scale=t['scale'],currency=t['currency_literal'],label=t['label']['text'],external=t['external_metadata'],unknown=t['unclassified_header_shapes_numbers_redacted'],headers=t['headers']) for t in e['statement_contexts']]),sort_keys=True))
