"""Pure exact 253-price / 12-block basket statistics. No I/O or square roots."""
from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Real
import re

SCHEMA='monthly_basket_core251_measure_v1'
BASKET_SCHEMA='monthly_basket_core251_basket_v1'
MEASURE_KEYS=('schema','price_count','block_count','momentum','block_returns')
BASKET_KEYS=('schema','block_count','portfolio_blocks','mean','joint_variance','diagonal_variance','supported','reason')


def require(ok,message):
    if not ok:raise ValueError(message)


def _sequence(x,n,label):
    require(not isinstance(x,(str,bytes,Mapping,set,frozenset)),label+' ordered sequence')
    try:y=list(x)
    except TypeError as exc:raise ValueError(label+' sequence') from exc
    require(len(y)==n and getattr(x,'ndim',1)==1,label+' exact length');return y


def _pair(x):
    x=Fraction(x);return dict(numerator=str(x.numerator),denominator=str(x.denominator))


def _rational(x):
    if isinstance(x,Fraction):return x
    require(type(x) is dict and set(x)=={'numerator','denominator'},'Canonical rational fields')
    a,b=x['numerator'],x['denominator']
    require(type(a) is str and type(b) is str and re.fullmatch(r'0|-?[1-9][0-9]*',a) is not None and re.fullmatch(r'[1-9][0-9]*',b) is not None,'Canonical integer strings')
    f=Fraction(int(a),int(b));require(_pair(f)==x,'Reduced rational');return f


def measure(prices253):
    values=_sequence(prices253,253,'Prices');p=[];errors=[]
    for j,x in enumerate(values):
        try:
            require(isinstance(x,Real) and not isinstance(x,bool) and type(x).__name__!='bool_',f'Price {j} real nonboolean')
            try:v=float(x)
            except (OverflowError,TypeError,ValueError) as exc:raise ValueError(f'Price {j} binary64') from exc
            require(math.isfinite(v) and v>0,f'Price {j} finite positive')
            p.append(Fraction.from_float(v))
        except ValueError as exc:errors.append(str(exc));p.append(None)
    require(not errors,'; '.join(errors))
    r=[p[21*(b+1)]/p[21*b]-1 for b in range(12)]
    return dict(schema=SCHEMA,price_count=253,block_count=12,momentum=_pair(p[-1]/p[0]-1),block_returns=[_pair(x) for x in r])


def basket(block_returns3x12):
    rows=_sequence(block_returns3x12,3,'Basket');rr=[]
    for row in rows:
        r=[_rational(x) for x in _sequence(row,12,'Blocks')]
        require(all(x>-1 for x in r),'Positive gross block return');gross=Fraction(1)
        for x in r:gross*=1+x
        require(gross>1,'Each constituent positive annual momentum');rr.append(r)
    q=[sum((r[b] for r in rr),Fraction(0))/3 for b in range(12)];mean=sum(q,Fraction(0))/12
    require(mean>0,'Positive basket mean invariant')
    vj=sum(((x-mean)**2 for x in q),Fraction(0))/11
    vd=Fraction(0)
    for r in rr:
        mu=sum(r,Fraction(0))/12;vd+=sum(((x-mu)**2 for x in r),Fraction(0))/99
    reason='positive_variances' if vj>0 and vd>0 else ('zero_both_variances' if vj==0 and vd==0 else ('zero_joint_variance' if vj==0 else 'zero_diagonal_variance'))
    return dict(schema=BASKET_SCHEMA,block_count=12,portfolio_blocks=[_pair(x) for x in q],mean=_pair(mean),joint_variance=_pair(vj),diagonal_variance=_pair(vd),supported=vj>0 and vd>0,reason=reason)


def compare_ratios(mean_a,var_a,mean_b,var_b):
    a,va,b,vb=map(_rational,(mean_a,var_a,mean_b,var_b));require(a>0 and b>0 and va>0 and vb>0,'Positive ratio inputs')
    d=a*a*vb-b*b*va;return (d>0)-(d<0)
