from common import *
for x in json.loads((HERE/'NFLX/DISCOVERY.json').read_text()):
 try:b,p,m=fetch('NFLX',x['url']);print('NFLX',len(b),x['url'],flush=True)
 except Exception as e:print('NFLX ERROR',str(e),flush=True)
