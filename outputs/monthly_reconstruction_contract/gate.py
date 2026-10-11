"""Explicit evidence gate; neither return targeting nor an automatic optimizer."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def contract():
    return json.loads((P / 'CONTRACT.json').read_text())


def assess(evidence, kind='global_attribution_hypothesis'):
    c = contract()
    required = list(c['admission_requirements'])
    if kind == 'executable_rule_hypothesis':
        required += c['rule_additional_requirements']
    elif kind != 'global_attribution_hypothesis':
        raise ValueError('Only full attribution or executable-rule hypotheses can be admitted')
    statuses = {k: evidence.get(k, 'unresolved') for k in required}
    if any(v not in {'passed', 'failed', 'unresolved'} for v in statuses.values()):
        raise ValueError('Evidence must be passed, failed, or unresolved')
    return dict(admitted=all(v == 'passed' for v in statuses.values()), kind=kind,
                failed=[k for k, v in statuses.items() if v == 'failed'],
                unresolved=[k for k, v in statuses.items() if v == 'unresolved'])


def admitted_transactions(orders, evidence, kind='global_attribution_hypothesis'):
    result = assess(evidence, kind)
    if not result['admitted']:
        raise ValueError('Blocked: full-trajectory reconstruction evidence is incomplete or failed')
    # Passing a reconstruction test does not establish the author's actual holdings.
    return dict(evidence=result, author_transactions_verified=False,
                transactions=orders, use='research hypothesis only')
