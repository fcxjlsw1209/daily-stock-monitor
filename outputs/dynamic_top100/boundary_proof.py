"""Research membership proof from sourced bounds, not exact market-cap ranks."""
import math
from universe import utc

def prove_membership(rows, expected_issuer_ids, session, decision_at, n=100):
    """Every eligible issuer needs a bound; unknown upper bound is infinity.

    Evidence must justify bounds for the specific date, in total-company USD.
    This only proves membership; it does not certify prices or trading inputs.
    """
    if not isinstance(n,int) or n<1:raise ValueError('Invalid member count')
    expected=set(expected_issuer_ids)
    ids=[r['issuer_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError('Incomplete or duplicate issuer coverage')
    if len(rows)<n:raise ValueError('Too few issuers')
    decision=utc(decision_at);bounds=[]
    for r in rows:
        if r['session']!=session:raise ValueError('Bound date mismatch')
        if r['cap_basis']!='total_company_equity_usd':raise ValueError('Invalid capitalization basis')
        lo=float(r['lower_usd']);hi=float(r['upper_usd']) if r['upper_usd'] is not None else math.inf
        if not math.isfinite(lo) or lo<0 or math.isnan(hi) or hi<lo:raise ValueError('Invalid bounds')
        unknown=(lo==0 and hi==math.inf)
        if not unknown:
            if not r.get('source') or not r.get('source_sha256') or r.get('bound_verified') is not True:raise ValueError('Unverified bounds')
            if utc(r['available_at'])>decision:raise ValueError('Future bound evidence')
        bounds.append((r['issuer_id'],lo,hi))
    # Strict separation is conservative: tied/overlapping boundary remains unresolved.
    ordered=sorted(bounds,key=lambda x:(-x[1],x[0]));chosen=ordered[:n];rest=ordered[n:]
    floor=min(x[1] for x in chosen);ceiling=max((x[2] for x in rest),default=-math.inf)
    proven=floor>ceiling
    unresolved=[x[0] for x in bounds if x[1]<=ceiling and x[2]>=floor] if not proven else []
    return {'session':session,'membership_proven':proven,'members':sorted(x[0] for x in chosen) if proven else [],'minimum_member_cap_usd':floor,'maximum_outsider_cap_usd':ceiling if math.isfinite(ceiling) else None,'boundary_issuer_ids':sorted(unresolved),'exact_ranks_known':False,'trading_inputs_certified':False}
