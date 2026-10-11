"""Resume one public 36.6MB archive; never treat partial files as downloaded data."""
from pathlib import Path
import sys,time,json,hashlib
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'vendor'))
import libtorrent as lt
folder=P/'sources/quantquote'
ti=lt.torrent_info(str(folder/'dataset.torrent'))
assert ti.total_size()==36616638 and ti.num_files()==1
assert ti.files().file_path(0)=='quantquote_daily_sp500_83986.zip'
ses=lt.session({'enable_dht':True,'enable_lsd':False,'enable_upnp':False,'enable_natpmp':False,'upload_rate_limit':1024})
h=ses.add_torrent({'ti':ti,'save_path':str(folder)})
start=time.monotonic();complete=False
try:
 while time.monotonic()-start<1800:
  s=h.status();state={'elapsed':round(time.monotonic()-start),'progress':round(s.progress,4),'peers':s.num_peers,'download_bytes':s.total_done,'error':str(s.errc) if s.errc.value() else None}
  (folder/'DOWNLOAD_STATUS.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
  if s.is_seeding:
   h.pause();f=folder/'quantquote_daily_sp500_83986.zip'
   (folder/'archive.provenance.json').write_text(json.dumps({'url':'https://academictorrents.com/details/49daf05ef35c487331013c22450988bbf7e511b0','sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'torrent_hash':'49daf05ef35c487331013c22450988bbf7e511b0','peer_piece_validation':True,'bytes':f.stat().st_size},indent=2)+'\n');complete=True;break
  if s.errc.value():break
  time.sleep(30)
finally:
 h.pause();ses.pause()
if not complete:raise SystemExit('Archive incomplete; partial pieces retained for verified resume')
