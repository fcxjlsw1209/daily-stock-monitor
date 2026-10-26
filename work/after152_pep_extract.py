"""Pure PEP FY2010–2016 source extraction; no file reads or admission.

The shared geometry routines accept only complete date-only headers. Thus a
prior-year audit date embedded in the income caption cannot become a column
label. PepsiCo's total NI is distinct from the attributable-to-PepsiCo row.
"""
import after152_mu_extract as common

FIELDS, STATUS = common.FIELDS, common.STATUS
SourceError, FieldGap, sha256 = common.SourceError, common.FieldGap, common.sha256
YEARS = tuple(range(2010, 2017))
PROFILE = dict(symbol='PEP', issuer='PepsiCo, Inc.', years=YEARS,
               hosts=('pepsico.mediaroom.com', 'www.prnewswire.com'),
               issuer_pattern=r'\bPepsiCo,?\s+Inc\.', ticker_pattern=r'\b(?:NYSE|NASDAQ)\s*:\s*PEP\b',
               month=12, income_groups=4, balance_groups=2,
               assumption_id='PEP_consecutive_annual_boundaries_v1')


def source_files():
    return tuple(dict.fromkeys(('work/after152_pep_extract.py', 'work/test_after152_pep_extract.py') + common.source_files()))


def extract_document(raw_bytes, context):
    return common._extract(raw_bytes, context, PROFILE)
