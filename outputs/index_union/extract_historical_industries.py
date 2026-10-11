"""Reviewed OEF industry observations, not a GICS or full-period certificate."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
GROUPS={
'AEROSPACE & DEFENSE':'BA GD LMT RTN UTX','AGRICULTURE':'MO MON PM','APPAREL':'NKE',
'AUTO MANUFACTURERS':'F','BANKS':'BAC BK RF USB WFC','BEVERAGES':'KO PEP','BIOTECHNOLOGY':'AMGN',
'CHEMICALS':'DOW DD','COMPUTERS':'AAPL DELL EMC HPQ IBM','COSMETICS & PERSONAL CARE':'AVP CL PG',
'DIVERSIFIED FINANCIAL SERVICES':'AXP COF C GS JPM MS NYX','ELECTRIC':'AEP ETR EXC SO',
'FOOD':'CPB HNZ KFT SLE','FOREST PRODUCTS & PAPER':'WY','HEALTH CARE - PRODUCTS':'BAX COV JNJ MDT',
'HEALTH CARE - SERVICES':'UNH','INSURANCE':'ALL','INTERNET':'AMZN GOOG','MACHINERY':'CAT',
'MANUFACTURING':'GE HON MMM','MEDIA':'CMCSA TWX DIS','MINING':'AA','OFFICE & BUSINESS EQUIPMENT':'XRX',
'OIL & GAS':'CVX COP DVN XOM OXY','OIL & GAS SERVICES':'BHI HAL NOV SLB',
'PHARMACEUTICALS':'ABT BMY GILD MRK PFE SGP WYE','PIPELINES':'WMB',
'RETAIL':'COST CVS HD LOW MCD TGT WAG WMT','SEMICONDUCTORS':'INTC TXN',
'SOFTWARE':'MA MSFT ORCL','TELECOMMUNICATIONS':'T CSCO QCOM S VZ','TRANSPORTATION':'BNI FDX NSC UPS'}

def main():
    source=P/'sources/oef_march2009.pdf'
    proof=json.loads((P/'sources/oef_march2009.provenance.json').read_text())
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if sha!=proof['sha256']:raise ValueError('Source changed')
    expected=set(json.loads((P/'sp100_2009_holdings_corrob.json').read_text())['tickers_inferred_from_printed_company_names'])
    symbols=[s for group in GROUPS.values() for s in group.split()]
    if len(symbols)!=100 or len(set(symbols))!=100 or set(symbols)!=expected:
        raise ValueError('Industry transcription does not match reviewed 100 holdings')
    rows=[dict(symbol_as_of_report=s,observed_industry=group,observed_as_of='2009-03-31',
               known_at='2009-06-05T23:59:59-04:00',source_file=str(source.relative_to(P)),
               source_sha256=sha,classification_scheme='issuer_report_industry_not_GICS',
               observation_verified=True,permanent_security_id=None,strategy_sector=None)
          for group,tickers in GROUPS.items() for s in tickers.split()]
    result=dict(source_url=proof['url'],source_representation='issuer_report_mirror',
                pdf_pages_one_based=[28,29,30],printed_pages=[23,24,25],
                publication_evidence='ASX cover letter dated 5 June 2009 on PDF page1',
                visual_review=True,observations=rows,observation_count=100,
                limitations=['Report industry groups are not GICS: Internet contains both Amazon and Google; Telecommunications contains both Cisco/Qualcomm and telecom operators.',
                             'Do not map a whole report group to one strategy sector without issuer-specific review.',
                             'Historical security identities, later sector transitions and NDX-only constituents still required.'])
    (P/'HISTORICAL_INDUSTRY_OBSERVATIONS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'historical_industry_observations':len(rows),'groups':len(GROUPS),'full_period_sector_certified':False}))

if __name__=='__main__':main()
