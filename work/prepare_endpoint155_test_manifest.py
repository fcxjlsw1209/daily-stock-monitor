"""Archive completed synthetic logs before round155 registration, no market reads."""
from pathlib import Path
import argparse
import re

import monthly_endpoint155_lifecycle as lifecycle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in lifecycle.SUITES:parser.add_argument('--'+name.replace('_','-')+'-log',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=root/'outputs'/lifecycle.STUDY
    if (out/'PLAN.json').exists():raise ValueError('Cannot replace logs after registration')
    out.mkdir(exist_ok=True);suites=[]
    for name,filename in lifecycle.SUITES.items():
        source=root/'work'/filename;raw=Path(getattr(args,name+'_log')).read_bytes();text=raw.decode()
        counts=re.findall(r'Ran ([0-9]+) tests? in ',text)
        if len(counts)!=1 or not text.rstrip().endswith('OK'):raise ValueError('Ambiguous or unsuccessful suite '+name)
        count=int(counts[0]);destination=out/(name.upper()+'_TESTS.log')
        if destination.exists():
            if destination.read_bytes()!=raw:raise ValueError('Archived log differs: '+name)
        else:lifecycle.publish_bytes(destination,raw)
        suites.append(dict(name=name,count=count,passed=True,
            source=dict(path=str(source.relative_to(root)),sha256=lifecycle.sha(source)),
            log=dict(path=str(destination.relative_to(root)),sha256=lifecycle.sha(destination))))
    manifest=dict(passed=True,synthetic_only=True,suites=suites,test_count=sum(x['count'] for x in suites))
    lifecycle.validate_tests(manifest,lifecycle.old.Reader(root))
    lifecycle.write(out/'TESTS.json',manifest)
    print(lifecycle.old.encoded(dict(test_count=manifest['test_count'],sha256=lifecycle.sha(out/'TESTS.json'))).decode())


if __name__=='__main__':main()
