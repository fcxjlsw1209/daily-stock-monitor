"""Explicit runtime-only VOX attachment for262. Never reads on import."""
from pathlib import Path
import hashlib,importlib.util,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PARSER=ROOT/'work/ownsector238_repair_sources.py'
PARSER_SHA='b88c0f750ea4e3d858543609083220d5f85ea026bc9d715cca63539b9accab5c'
def need(ok,msg):
    if not ok:raise ValueError(msg)
TRANSITIVE=ROOT/'work/ownsector238_source_audit.py'
TRANSITIVE_SHA='7d6eb858f9aeacd1a8ae627f8d41ba9f813a4b6c1bf95472fff98187e607b9b4'
def dependency_files():return (PARSER,TRANSITIVE)
def _path(raw):
    p=Path(raw);p=p if p.is_absolute() else ROOT/p;p=p.resolve();need(p.is_relative_to(ROOT),'Local approved source only');return p
def attach_vox(panel,review,calendar):
    need(isinstance(panel,dict) and 'adj' in panel and isinstance(panel['adj'],pd.DataFrame),'Original adjusted panel')
    need('VOX' not in panel['adj'].columns,'No source replacement')
    need(review.get('schema')=='tracking_risk262_source_contract_v1' and review.get('source_admitted') is False and review.get('metadata_only') is True,'Prospective source contract')
    need(hashlib.sha256(PARSER.read_bytes()).hexdigest()==PARSER_SHA==review['vox']['parser_sha256'],'Frozen parser')
    need(hashlib.sha256(TRANSITIVE.read_bytes()).hexdigest()==TRANSITIVE_SHA,'Frozen parser import dependency')
    p=_path(review['vox_prices_path']);raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==review['vox_prices_sha256'],'Dedicated approved VOX source')
    spec=importlib.util.spec_from_file_location('tracking262_vox_parser238',PARSER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    vox=module.parse_vox(json.loads(raw),calendar)
    adj=panel['adj'].copy();adj['VOX']=vox['VOX'].reindex(adj.index)
    return {'adj':adj}
