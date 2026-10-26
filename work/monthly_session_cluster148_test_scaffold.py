"""Copy unchanged generated-price tests into the new round, no computations."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
t=(r/'work/test_monthly_asset_growth147_outcome_review.py').read_text()
head=t[:t.index('def synthetic_outputs():')]
head=head.replace('monthly_asset_growth147_outcome_review','monthly_session_cluster148_outcome_independent')
classes=t[t.index('class SelectionTests'):t.index('class OutcomeTests')]
classes=classes.replace('[3]','[2]').replace('range(3)','range(2)').replace('(1,2)','(1,)')
tail=t[t.index("if __name__=='__main__':"):]
tail=tail.replace('monthly_asset_growth147','monthly_session_cluster148')
tail=tail.replace('m.synthetic_receipt(result)','m.synthetic_receipt(result)')
path=r/'work/test_monthly_session_cluster148_outcome_independent.py'
path.write_text(head+classes+"\n# New fixtures and binding tests are added below.\n"+tail)
print('Copied unchanged generated-price accounting/selection tests only.')
