"""Read-only pre-registration exercise of real source metadata, without factors."""
from pathlib import Path
import json
from complete_monthly_same_report145 import Reader, source_closure
root=Path(__file__).resolve().parents[1]
reader=Reader(root)
sources=source_closure(reader,root/'outputs/monthly_same_report145_union/AUDIT.json',root/'outputs/monthly_same_report145_union/DATA.json')
print(json.dumps({'passed':True,'source_files':len(sources),'new_factor_or_return_calculations':0}))
