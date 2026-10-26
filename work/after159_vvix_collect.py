import hashlib,json,pathlib,datetime,urllib.request,urllib.error,sys
ROOT=pathlib.Path('/Users/shiweili/code/monthly-stock-research')
PLAN=ROOT/'work/after159_vvix_source_plan.json'
plan=json.loads(PLAN.read_text())
OUT=ROOT/plan['output_directory'];OUT.mkdir(parents=True,exist_ok=True)
MAN=OUT/'COLLECTION.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if MAN.exists():
    manifest=json.loads(MAN.read_text())
    assert manifest['plan_sha256']==sha(PLAN.read_bytes())
else:manifest={'plan_sha256':sha(PLAN.read_bytes()),'started_utc':utc(),'sources':{}}
if manifest.get('stop_for_429'):
    raise SystemExit('Prior429 recorded; no further request allowed by this collection.')
names={'historical_index_page':'historical_index_page.html','VVIX':'VVIX_History.csv','VIX':'VIX_History.csv','methodology':'methodology.pdf','launch_release':'launch_release.html'}
for key,url in plan['sources'].items():
    path=OUT/names[key]
    prior=manifest['sources'].get(key)
    if prior and prior.get('status')==200:
        assert path.is_file() and sha(path.read_bytes())==prior['sha256']
        continue
    if prior:raise SystemExit('Prior non-success preserved; no automatic retry: '+key)
    assert not path.exists(), 'Unbound existing source file'
    record={'requested_url':url,'retrieved_utc':utc(),'file':str(path.relative_to(ROOT))}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; historical-research/1.0)'})
        with urllib.request.urlopen(req,timeout=40) as response:
            body=response.read(10*1024*1024+1)
            assert len(body)<=10*1024*1024,'Response size limit'
            record.update(status=response.status,final_url=response.geturl(),headers={k:response.headers.get(k) for k in ['Content-Type','Last-Modified','ETag','Date','Retry-After']},sha256=sha(body),bytes=len(body))
        with path.open('xb')as f:f.write(body)
    except urllib.error.HTTPError as e:
        record.update(status=e.code,final_url=e.geturl(),retry_after=e.headers.get('Retry-After'),error=str(e))
        if e.code==429:manifest['stop_for_429']=True
    except Exception as e:record.update(status=None,error=type(e).__name__+': '+str(e))
    manifest['sources'][key]=record;manifest['updated_utc']=utc()
    tmp=OUT/'COLLECTION.tmp';tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');tmp.replace(MAN)
    print(json.dumps({'source':key,**{k:record.get(k)for k in ('status','bytes','sha256','error')}}),flush=True)
    if record.get('status')!=200:break
manifest['complete']=len(manifest['sources'])==len(names)and all(r.get('status')==200 for r in manifest['sources'].values())
tmp=OUT/'COLLECTION.tmp';tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');tmp.replace(MAN)
if not manifest['complete']:sys.exit(1)
