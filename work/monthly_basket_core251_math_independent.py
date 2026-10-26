"""Independent251 exact basket algebra; pure synthetic/source-window functions.

Variance uses second-moment identities, independently of centered two-pass math.
No producer, source loader, filesystem, network or account engine is imported.
"""
from fractions import Fraction
import math
from numbers import Real
import re
import numpy as np

MEASURE_SCHEMA='monthly_basket_core251_measure_v1'
BASKET_SCHEMA='monthly_basket_core251_basket_v1'
MEASURE_KEYS=('schema','price_count','block_count','momentum','block_returns')
BASKET_KEYS=('schema','block_count','portfolio_blocks','mean','joint_variance','diagonal_variance','supported','reason')


def pair(value):
    if type(value) is not Fraction:raise ValueError('Fraction required')
    return dict(numerator=str(value.numerator),denominator=str(value.denominator))


def rational(value):
    if type(value) is Fraction:return value
    if type(value) is not dict or set(value)!={'numerator','denominator'}:raise ValueError('Canonical rational pair required')
    a,b=value['numerator'],value['denominator']
    if type(a) is not str or type(b) is not str or not re.fullmatch(r'-?(0|[1-9][0-9]*)',a) or a=='-0' or not re.fullmatch(r'[1-9][0-9]*',b):raise ValueError('Canonical rational strings')
    out=Fraction(int(a),int(b))
    if pair(out)!=value:raise ValueError('Reduced rational required')
    return out


_pair,_rational=pair,rational


def exact(actual,expected,label='audit'):
    if type(actual) is not type(expected):raise ValueError(label+' type')
    if type(expected) is dict:
        if set(actual)!=set(expected):raise ValueError(label+' keys')
        for k,v in expected.items():exact(actual[k],v,label+'/'+str(k))
    elif type(expected) in (list,tuple):
        if len(actual)!=len(expected):raise ValueError(label+' length')
        for j,(a,e) in enumerate(zip(actual,expected)):exact(a,e,label+'/'+str(j))
    elif type(expected) is float:
        if actual.hex()!=expected.hex():raise ValueError(label+' float bits')
    elif actual!=expected:raise ValueError(label+' value')


def measure_oracle(prices253):
    a=np.asarray(prices253,dtype=object)
    if a.shape!=(253,):raise ValueError('253 one-dimensional prices required')
    p=[];errors=[]
    for i,x in enumerate(a.tolist()):
        if not isinstance(x,Real) or isinstance(x,(bool,np.bool_)):
            errors.append((i,'real_nonboolean'));continue
        try:y=float(x)
        except (TypeError,ValueError,OverflowError):errors.append((i,'conversion'));continue
        if not math.isfinite(y) or y<=0.:errors.append((i,'positive_finite'));continue
        p.append(Fraction.from_float(y))
    if errors:raise ValueError('Invalid required prices '+repr(errors))
    gains=[(p[21*(b+1)]-p[21*b])/p[21*b] for b in range(12)]
    momentum=(p[-1]-p[0])/p[0]
    product=Fraction(1)
    for r in gains:product*=1+r
    if product!=1+momentum:raise ValueError('Exact block clock telescoping')
    return dict(schema=MEASURE_SCHEMA,price_count=253,block_count=12,
      momentum=pair(momentum),block_returns=[pair(r) for r in gains])


def basket_oracle(block_returns3x12):
    if not isinstance(block_returns3x12,(list,tuple)) or len(block_returns3x12)!=3:raise ValueError('Three block series required')
    series=[]
    for row in block_returns3x12:
        if not isinstance(row,(list,tuple)) or len(row)!=12:raise ValueError('Twelve blocks required')
        rs=[rational(x) for x in row];gross=Fraction(1)
        for r in rs:
            if r<=-1:raise ValueError('Positive block gross returns')
            gross*=1+r
        if gross<=1:raise ValueError('Positive annual momentum required')
        if sum(rs,Fraction())<=0:raise ValueError('AM-GM positive mean invariant')
        series.append(rs)
    q=[sum((row[b] for row in series),Fraction())/3 for b in range(12)]
    total=sum(q,Fraction());mean=total/12
    # Difference from producer: raw second moments, not squared centered values.
    joint=(sum((x*x for x in q),Fraction())-total*total/12)/11
    diagonal=sum((sum((r*r for r in row),Fraction())-sum(row,Fraction())**2/12 for row in series),Fraction())/99
    if mean<=0 or joint<0 or diagonal<0 or joint>3*diagonal:raise ValueError('Exact mean/variance invariant')
    supported=joint>0 and diagonal>0
    reason=('positive_variances' if supported else 'zero_both_variances' if joint==diagonal==0 else 'zero_joint_variance' if joint==0 else 'zero_diagonal_variance')
    return dict(schema=BASKET_SCHEMA,block_count=12,portfolio_blocks=[pair(x) for x in q],
      mean=pair(mean),joint_variance=pair(joint),diagonal_variance=pair(diagonal),supported=supported,reason=reason)


def compare_ratios(mean_a,var_a,mean_b,var_b):
    a,x,b,y=[rational(v) for v in (mean_a,var_a,mean_b,var_b)]
    if min(a,x,b,y)<=0:raise ValueError('Positive means and variances required')
    # Compare squares after positive cross-multiplication; no floating scores.
    difference=a*a*y-b*b*x
    return (difference>0)-(difference<0)


def verify_measure(prices,actual):
    exact(actual,measure_oracle(prices));return dict(passed=True,exact_rational=True,price_count=253,block_count=12)


def verify_basket(blocks,actual):
    exact(actual,basket_oracle(blocks));return dict(passed=True,exact_rational=True,common_variance_domain_checked=True)


measure=measure_oracle
basket=basket_oracle
