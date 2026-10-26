"""Pure sector-local versus global mid-CDF on frozen peer239 source support.

The frozen producer owns source/clock/maturity validation and descriptive
momentum/median evidence. Only the two target arrays and their audit values
change. No data loading, fitting, selection or portfolio replay.
"""
from pathlib import Path
import hashlib
import importlib.util
import math

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'work/monthly_peer239_kernel.py'
FROZEN_SHA256 = '9d0b9cc7d39243a6d9ca3117a71b97c93c2e1be4adb67757ff5cfaa42f0da419'
AXES = ROOT/'work/monthly_relative_gate235_kernel.py'
AXES_SHA256 = 'bab4a2b9794ef00a63d43422a51cef167daffbb11b31dc0452cfcd2e74a6af8d'
VALUE_KEYS = ('mature','supported','momentum','peer_median','peer_target','absolute_target')
COUNT_FIELDS = ('events','stock_cells','mature_cells','supported_cells','endpoint_observations','new_model_fits')
EVENT_KEYS = ('query_index','date','source_dates','source_positions','stock_indices','mature_indices','supported_indices','cells')
CELL_KEYS = ('column_index','symbol','issuer_key','sector','history_status','peer_indices','peer_issuer_count','momentum','peer_median','peer_difference','peer_target','absolute_target','peer_numerator','peer_denominator','absolute_numerator','absolute_denominator')
INTEGER_FIELDS = CELL_KEYS[-4:]


def require(ok, message):
    if not ok: raise ValueError(message)


def frozen_helper():
    for path, digest in ((FROZEN,FROZEN_SHA256),(AXES,AXES_SHA256)):
        require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,
                'Frozen source/axes dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location('peer_rank240_frozen239',FROZEN)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_files():
    return (FROZEN,AXES)


def _midcdf(value, domain):
    require(bool(domain) and math.isfinite(value) and all(math.isfinite(x) for x in domain),
            'Finite nonempty rank domain required')
    less=sum(x<value for x in domain)
    equal=sum(x==value for x in domain)
    require(equal>=1,'Rank domain must include the focal observation')
    numerator=2*less+equal-len(domain)
    denominator=2*len(domain)
    target=float(numerator)/float(denominator)
    if not math.isfinite(target): raise FloatingPointError('Nonfinite derived mid-CDF')
    return target,numerator,denominator


def build_features(panel,meta,calendar,arrays,source_report,limit=None):
    values,audit=frozen_helper().build_features(panel,meta,calendar,arrays,source_report,limit=limit)
    require(tuple(values)==VALUE_KEYS and tuple(audit['counts'])==COUNT_FIELDS,
            'Frozen feature/count interface changed')
    require(audit['schema']=='monthly_peer239_features_v1','Frozen feature schema changed')
    for event in audit['events']:
        require(tuple(event)==EVENT_KEYS,'Frozen event interface changed')
        i=event['query_index'];supported=event['supported_indices']
        global_domain=[float(values['momentum'][i,j]) for j in supported]
        for cell in event['cells']:
            require(tuple(cell)==CELL_KEYS[:-4],'Frozen cell interface changed')
            j=cell['column_index']
            for key in INTEGER_FIELDS:cell[key]=None
            if j not in supported:continue
            local_indices=sorted([j]+cell['peer_indices'])
            require(len(local_indices)>=3 and len(set(local_indices))==len(local_indices)
                    and all(k in supported for k in local_indices),'Complete supported sector domain required')
            local_domain=[float(values['momentum'][i,k]) for k in local_indices]
            momentum=float(values['momentum'][i,j])
            for prefix,key,domain in (('peer','peer_target',local_domain),('absolute','absolute_target',global_domain)):
                target,numerator,denominator=_midcdf(momentum,domain)
                values[key][i,j]=target;cell[key]=target
                cell[prefix+'_numerator']=numerator;cell[prefix+'_denominator']=denominator
    audit['schema']='monthly_peer_rank240_features_v1'
    return values,audit
