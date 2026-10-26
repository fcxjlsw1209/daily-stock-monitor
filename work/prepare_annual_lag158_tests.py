"""Archive actual synthetic test logs without running market research."""
from pathlib import Path
import argparse
import re
import monthly_annual_lag158_lifecycle as lifecycle
from monthly_endpoint155_lifecycle import publish_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in lifecycle.SUITES:parser.add_argument('--'+name.replace('_','-')+'-log',required=True)
    args=parser.parse_args();root=lifecycle.ROOT;out=root/lifecycle.OUT
    if (out/'PLAN.json').exists():raise ValueError('Already registered; preserve test evidence')
    out.mkdir(exist_ok=True);suites=[]
    for name,filename in lifecycle.SUITES.items():
        source=root/'work'/filename;raw=Path(getattr(args,name+'_log')).read_bytes();text=raw.decode()
        counts=re.findall(r'Ran (\d+) tests? in ',text)
        if len(counts)!=1 or not text.rstrip().endswith('OK'):raise ValueError('Not one successful suite: '+name)
        count=int(counts[0]);destination=out/(name.upper()+'_TESTS.log')
        if destination.exists():
            if destination.read_bytes()!=raw:raise ValueError('Existing log differs')
        else:publish_bytes(destination,raw)
        suites.append(dict(name=name,count=count,passed=True,
            source=dict(path=str(source.relative_to(root)),sha256=lifecycle.sha(source)),
            log=dict(path=str(destination.relative_to(root)),sha256=lifecycle.sha(destination))))
    doc=dict(passed=True,synthetic_only=True,suites=suites,test_count=sum(x['count'] for x in suites))
    lifecycle.validate_tests(doc,lifecycle.old.Reader(root));lifecycle.write(out/'TESTS.json',doc)
    print(lifecycle.old.encoded(dict(test_count=doc['test_count'],sha256=lifecycle.sha(out/'TESTS.json'))).decode())


if __name__=='__main__':main()
