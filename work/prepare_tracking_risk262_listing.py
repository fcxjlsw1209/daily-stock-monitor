"""Metadata-only early actual-trade evidence; no network or price decoding."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
OUT=R/'work/tracking_risk262_listing_policy.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def main():
    need(not OUT.exists(),'write once')
    d=R/'outputs/monthly_tracking_risk262_sources'
    rendering=d/'spcx_ipo_closing_20260615.web_rendering.json'
    need(sha(rendering)=='58251850e4d748adf1768f15eb9b5f177471c145b03e217e78d78eb46c8ac67b','rendering digest')
    r=json.loads(rendering.read_text());t=r['result']
    need('June 15, 2026' in t and 'shares of Class A common stock began trading' in t and 'June\u00a012, 2026' in t and '“SPCX.”' in t,'official issuer/date/actual trade claim')
    oldpath=R/'work/monthly_peer_path252_v2_LISTING_POLICY.json';old=json.loads(oldpath.read_text())
    pins={str(oldpath.relative_to(R)):sha(oldpath),str(rendering.relative_to(R)):sha(rendering)}
    acquisition=[]
    for stem in ('spcx_ipo_closing_20260615','spcx_ipo_closing_canonical_20260615','nasdaq_spcx_ipo_20260612','nasdaq_spcx_listing_20260616'):
        p=d/(stem+'.retrieval.json');o=json.loads(p.read_text());suffix='.html' if o['status']==200 else '.http_error';body=d/(stem+suffix)
        need(sha(body)==o['sha256'] and body.stat().st_size==o['bytes'],'acquisition binding')
        for f in (p,body):pins[str(f.relative_to(R))]=sha(f)
        acquisition.append(dict(path=str(body.relative_to(R)),sha256=sha(body),receipt_path=str(p.relative_to(R)),receipt_sha256=sha(p),http_status=o['status'],scope='corroboration only' if 'nasdaq_' in stem else 'HTTP403 failure; not official page content'))
    for p,h in old['input_sha256'].items():
        pp=Path(p);pp=pp if pp.is_absolute() else R/pp;need(sha(pp)==h,'frozen old evidence');pins[str(pp.relative_to(R))]=h
    pins[str(Path(__file__).relative_to(R))]=sha(Path(__file__))
    bound=dict(old['path_bound_override']);bound.update(available_from='2026-06-16',evidence_id='spcx_issuer_ipo_closing_20260615_actual_trade')
    out=dict(schema='tracking_risk262_early_listing_policy_v1',created_utc=datetime.now(timezone.utc).isoformat(),source_only=True,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,
      issuer='Space Exploration Technologies Corp.; Nasdaq Class A common stock; CUSIP84615Q103; not the previous SPCX fund',publication_date='2026-06-15',original_history_bound=old['original_history_bound'],path_bound_override=bound,
      late_policy_preserved=dict(path=str(oldpath.relative_to(R)),sha256=sha(oldpath),available_from='2026-08-05'),
      current_visible_queries=[dict(query_index=q,date=dt) for q,dt in ((119,'2026-06-30'),(120,'2026-07-31'),(121,'2026-08-31'))],
      official_evidence=dict(path=str(rendering.relative_to(R)),sha256=sha(rendering),url=r['url'],format=r['format'],locator='rendered lines7-9',publication_date='2026-06-15',claim='Issuer reports that its Class A shares began trading June12 under SPCX.'),
      acquisition=acquisition,
      policy=dict(identity='Apply only exact symbol SPCX AND original whole issuer key AND Class A identity; original_history_bound must equal the frozen six-key old239 bound.',visibility='Only query>=2026-06-16, declared official publication date plus next calendar day. No application to earlier queries.',source_validation='Required slots before visible path bound permit only ordinary missing; present malformed/zero/negative/nonfinite/bool or positive price is hard failure. At/after bound every required slot must be positive finite real nonbool. No fill, splice or pool shrinking.',maturity='Original history bound and structural warmup classification unchanged; no mapping repair or new history created.',known_gap_resolution='The earlier official statement resolves the June/July timing-definition blocker for four known June8-11 empty slots. It does not prove actual 262 numeric source admission.'),
      limitations=['Web-tool extracted rendering, not raw HTTP HTML, not an original June historical archive; content revision history not certified.','Both direct official URL requests returned HTTP403; those failures are preserved.','Current Nasdaq June12 HTML has dateModified July10 and is corroboration only, not June12-vintage proof. Nasdaq June16 current HTML is also not a historical archive.','No proof of65 required price slots, full coverage, sector mapping, price adjustment/unit continuity or strict point-in-time data.','No old239/252 policy, source, experiment or observer is modified.'],
      prices_loaded=False,npz_decoded=False,features_orders_returns_computed=False,strategy_registered=False,input_sha256=dict(sorted(pins.items())))
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(OUT.relative_to(R)),sha256=sha(OUT),metadata_only=True,source_admitted=False)))
if __name__=='__main__':main()
