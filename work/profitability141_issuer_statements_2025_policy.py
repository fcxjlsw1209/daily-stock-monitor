from bs4 import BeautifulSoup
import profitability141_issuer_statements as m
s=BeautifulSoup((m.ROOT/'outputs/monthly_earnings_repair/microsoft/raw/FY2025Q4.html').read_bytes(),'html.parser');t=m.norm(s.get_text(' ',strip=True))
import re,json
print(json.dumps(re.findall(r'.{0,100}(?:GAAP|generally accepted).{0,150}',t)))
