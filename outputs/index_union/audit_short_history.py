"""Bound the entry impact of new spin-off membership disputes.

This is not a constituent certificate and does not waive parent entitlements.
Even counting when-issued sessions, fewer than 253 sessions cannot pass the
unchanged six strategies' common warmup. Never splice a parent's price series.
"""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
from research_calendar import sessions
P=Path(__file__).resolve().parent


def assess(first_possible_session, through, minimum=253):
    calendar=sessions(first_possible_session,through)
    n=len(calendar)
    return dict(first_possible_session=first_possible_session,through=through,
        maximum_possible_observations=n,required_consecutive_observations=minimum,
        can_pass_history_requirement=n>=minimum,
        new_entry_impact_proven_zero=n<minimum,
        corporate_action_entitlements_waived=False)


def main():
    sources=json.loads((P/'SHORT_HISTORY_SOURCE_FACTS.json').read_text())
    rows=[]
    for source in sources['securities']:
        for receipt in source['sources']:
            file=P/receipt['file']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=receipt['sha256']:
                raise ValueError('Changed spin-off source: '+str(file))
        rows.append(dict(symbol=source['symbol'],**assess(source['first_possible_session'],sources['through']),sources=source['sources']))
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),through=sources['through'],
        rows=rows,membership_dates_certified=False,backtest_admitted=False,
        scope='Only new-entry eligibility. Dates are conservative earliest when-issued dates, not claimed index entry dates. Parent holdings and distributions remain separate obligations.',
        equations_sha256=hashlib.sha256((P/'dynamic_features.py').read_bytes()).hexdigest())
    (P/'SHORT_HISTORY_IMPACT.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows: print(row['symbol'],row['maximum_possible_observations'],row['new_entry_impact_proven_zero'])

if __name__=='__main__':main()
