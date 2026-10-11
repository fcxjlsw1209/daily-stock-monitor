"""Retain monotone core momentum; limit nonlinear correction to side factors."""
from itertools import product
import threshold_probe as t
from decisions import P,CORE,FEATURES,dump

if __name__=='__main__':
    t.STUMPS=list(product(range(len(CORE),len(FEATURES)),[-.25,0.,.25]))
    t.NAMES=FEATURES+[f'{FEATURES[j]}>{v:+.2f}' for j,v in t.STUMPS]
    original_dump=t.dump
    def scoped_dump(name,value):
        if name.startswith('THRESHOLD_'):name='SIDE_'+name
        original_dump(name,value)
    t.dump=scoped_dump
    t.main()
