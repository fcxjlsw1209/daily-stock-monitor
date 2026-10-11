import json,shutil
from prepare import P,prepare
from vendor_pool import apply
import run as runner
import validate_selected as validator
O=P/'vendor32_research'
def main():
 m,d=prepare();d=apply(m,d,O)
 for n in ['prepare.py','factors.py','run.py','validate_selected.py','vendor_pool.py','run_vendor32.py','collect_vendor_caps.py']:shutil.copyfile(P/n,O/n)
 runner.P=O;runner.prepare=lambda:(m,d);runner.main()
 validator.P=O;validator.prepare=lambda:(m,d);validator.main()
 f=O/'validated_round1/CONFIG.json';x=json.loads(f.read_text());x['stocks']='Historical index-union dailyselectionTop32 by latest prior vendor marketcap snapshot, not complete daily cap; see POOL_AUDIT.json';f.write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
