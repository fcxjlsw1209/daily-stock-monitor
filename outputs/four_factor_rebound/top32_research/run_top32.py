"""Isolated rerun; original daily observer and old research outputs untouched."""
import json,shutil
import run as runner
import validate_selected as validator
from prepare import prepare,P
from top32_pool import apply_pool
OUT=P/'top32_research'
def main():
 m,d=prepare();d=apply_pool(m,d,OUT)
 for name in ["prepare.py","factors.py","run.py","validate_selected.py","top32_pool.py","run_top32.py"]:shutil.copyfile(P/name,OUT/name)
 original_prepare=runner.prepare
 runner.prepare=lambda:(m,d)
 # Existing runner writes P/validated_round1; separate namespace keeps old artifacts intact.
 runner.P=OUT;runner.main()
 validator.P=OUT;validator.prepare=lambda:(m,d);validator.main()
 p=OUT/'validated_round1/CONFIG.json';config=json.loads(p.read_text());config['stocks']='Estimated daily historical NDX union SP100 Top32 using available lagged SEC share disclosures; incomplete cap/class coverage, not exact certifiedTop32';config['pool_audit']='../POOL_AUDIT.json';p.write_text(json.dumps(config,indent=2)+'\n')
if __name__=='__main__':main()
