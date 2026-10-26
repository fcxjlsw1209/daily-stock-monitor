import collect as c
exec((c.HERE/'priority_add3.py').read_text().split('headers=c.RAW')[0])
for name in ['sap_early1.json','sap_early2.json']:
 src=c.RAW/name
 for v in c.load(src,{})['results']:
  if v['status']!='fulfilled':continue
  d=v['value'];tx=d['response']
  if len(tx)<1000:continue
  indexed('SAP',d['url'],tx,[src],f"Q{d['q']} {d['year']} original quarterly statement")
p=c.HERE/'build.py';s=p.read_text().replace("'SAP':[r'Press Release by SAP", "'SAP':[r'Walldorf,\\s*Germany\\s*[-,]+\\s*('+MDY+')',r'Press Release by SAP");p.write_text(s)
