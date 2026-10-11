"""Materialize reviewed corporate-event rows for chain diagnostics, not admission."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib
P=Path(__file__).resolve().parent
rows=[]
def add(day,action,sym,pub,source,kind='corporate_action'):
    f=P/source
    rows.append(dict(effective_session=day,action=action,ticker_as_announced_or_effective=sym,
      published_at=pub if 'T' in pub else datetime.fromisoformat(pub).replace(hour=23,minute=59,second=59,tzinfo=ZoneInfo('America/New_York')).isoformat(),
      source_file=source,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),event_kind=kind,
      effective_convention='prior_to_open',announcement_content_reviewed=True,identity_verified=False,chain_verified=False))
def pair(day,old,new,pub,source,kind='symbol_change'):
    add(day,'deletion',old,pub,source,kind);add(day,'addition',new,pub,source,kind)
pair('2010-02-16','BNI','BRK.B','2010-02-16','sources/brk_bni_completion2010_receipt.json','index_replacement_secondary_date_corroboration')
pair('2012-10-02','KFT','MDLZ','2012-09-25','sources/sp_kft2012.html','spinoff_parent_retained')
pair('2013-07-01','NWSA','FOXA','2013-06-18','SECURITY_ALIAS_EVIDENCE_04.json','spinoff_parent_retained')
pair('2014-04-03','GOOG','GOOGL','2014-03-11','sources/sp_releases/993f1e646f91dc9c.html','class_a_symbol_change')
add('2014-04-03','addition','GOOG','2014-03-11','sources/sp_releases/993f1e646f91dc9c.html','new_class_c_distribution')
pair('2014-12-31','WAG','WBA','2014-12-31','SECURITY_ALIAS_EVIDENCE.json','holding_company_reorganization')
pair('2015-06-15','ACT','AGN','2015-06-12','SECURITY_ALIAS_EVIDENCE_05.json')
for sym in ['CMCSK','FOX']:add('2015-09-21','addition',sym,'2015-08-10','sources/sp_releases/cf8e2e2b95e07d9f.html','additional_class')
add('2015-12-14','deletion','CMCSK','2015-12-10','SECURITY_ALIAS_EVIDENCE_02.json','class_conversion')
pair('2017-09-01','DOW','DWDP','2017-08-24','sources/sp_releases/19dbaf92eaab5b51.html','merger_successor')
pair('2018-02-27','PCLN','BKNG','2018-02-21','SECURITY_ALIAS_EVIDENCE_02.json')
pair('2019-03-19','FOXA','TFCFA','2019-03-13','sources/fox_symbol_reuse2019.html','old_issuer_symbol_change')
pair('2019-03-19','FOX','TFCF','2019-03-13','sources/fox_symbol_reuse2019.html','old_issuer_symbol_change')
pair('2019-06-03','DWDP','DD','2019-06-03T06:00:00-04:00','sources/dupont_regular2019.html','spinoff_parent_reverse_split')
pair('2020-04-06','UTX','RTX','2020-03-31','sources/sp_releases/8f3d960ddf91164e.html','spinoff_merger_parent')
pair('2022-06-09','FB','META','2022-05-31','SECURITY_ALIAS_EVIDENCE_02.json')
pair('2026-05-21','BK','BNY','2026-05-11','sources/bny_symbol2026.html')
(P/'sp100_supplemental_reviewed_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
print('supplemental',len(rows))
