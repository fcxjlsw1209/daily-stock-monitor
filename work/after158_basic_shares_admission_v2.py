"""Narrow v2 source repair: verified uniform table units, no ratio calculation."""
import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
import re
import after158_basic_shares_admission as old

ROOT=old.ROOT
SELF='work/after158_basic_shares_admission_v2.py'
TEST='work/test_after158_basic_shares_admission_v2.py'
POLICY='work/after158_basic_shares_admission_policy_v2.json'
OUT='outputs/after158_basic_shares_admission_v2'
V1_PLAN=old.OUT+'/PLAN.json'
FAILURE=old.OUT+'/FAILURE.json'
OLD_PINS={old.SELF:'4932231edba569534e6f6f0161e8fae819d8eeff9fd328acb1ebf279fa96ae9c',old.TEST:'394e9b84ab5aa24c7aa07781595720b1eb7524572c637c10530c8ef4c1ab6c76',V1_PLAN:'ae8d89a9ec520e3492dd72ecc4f6bcdd48657e5edc25b4720ec8a89e28adf8f8',old.POLICY:'37963c102a37588af91dff3ad43efbea3f84e136817216729cfdf40477c2648d'}
GENERIC_YEARS={'AMD':frozenset((2015,2016,2017,2018,2020,2021,2022,2023,2024,2025)),'MSFT':frozenset(range(2015,2027)),'NVDA':frozenset(range(2015,2027))}
SCALES={'thousands':1000,'millions':1000000,'billions':1000000000}
COUNTS_POLICY='Both original values must be finite strictly positive exact decimal strings with BASIC semantics. Accept the original independently verified share-specific units, or the fixed 34 AMD/MSFT/NVDA events\' unchanged uniform formal-table units only when nonempty original unit/locator evidence agrees with the scale, displayed-to-decimal-to-shares normalization is exact, and the selected same-accession GAAP Basic cell is exactly equal with matching original-value strings. Rounding-interval compatibility alone cannot establish generic table units. Never change share_specific=false to true. No use of missing as zero.'
require,read,sha,write,verify,merge,stamp,positive,period_ok=(old.require,old.read,old.sha,old.write,old.verify,old.merge,old.stamp,old.positive,old.period_ok)
map_row,summary=old.map_row,old.summary


def policy():
 p=deepcopy(read(old.POLICY));p['schema']='basic_shares_conditional_source_policy_v2';p['counts']=COUNTS_POLICY;return p


def prepare():
 verify(OLD_PINS)
 require(not any(old.path(old.OUT+'/'+n).exists()for n in ('EVENTS.json','ROWS.json','MANIFEST.json')),'Old failed run unexpectedly has actual outputs')
 failure={'schema':'basic_shares_admission_v1_failure','status':'FAILED_BEFORE_ANY_EVENT_OR_ROW_OUTPUT','exit_code':1,'error':'ValueError: Share-specific unit required','stack_locations':{'event_overlay':86,'run':150,'main':165},'evidence':{'kind':'parent_report_of_actual_tool_transcript','tool':'tools.write_stdin','session_id':15488,'chunk_id':'a47b5b','shell_log_available':False,'shell_log_sha256':None},'input_hashes':OLD_PINS,'actual_ratio_score_return_calculations':0,'old_outputs_absent':['EVENTS.json','ROWS.json','MANIFEST.json']}
 if old.path(FAILURE).exists():require(read(FAILURE)==failure,'Existing failure receipt differs')
 else:write(FAILURE,failure)
 write(POLICY,policy());print({'failure_sha256':sha(FAILURE),'policy_sha256':sha(POLICY)})


def validate_unit(cell,bridge_cell,selected_context,raw,unique):
 flag=cell['unit']['share_specific']
 if flag is True:return  # Identical v1 rule for the ten direct-share-unit events.
 require(flag is False,'Share-specific unit flag must be boolean')
 require(raw['symbol']in GENERIC_YEARS and raw['fiscal_year']in GENERIC_YEARS[raw['symbol']] and raw['event_id'].startswith(raw['symbol']+':FY'+str(raw['fiscal_year'])+':'),'Outside fixed generic-unit event scope')
 u=cell['unit'];name=u.get('unit');scale=u.get('scale')
 require(name in SCALES and type(scale)is int and scale==SCALES[name],'Uniform unit name/scale mismatch')
 evidence=u.get('evidence');require(type(evidence)is list and evidence,'Uniform unit original evidence required')
 for item in evidence:
  require(type(item)is dict and type(item.get('text'))is str and type(item.get('locator'))is dict and item['locator'],'Original unit literal/locator required')
  literal=item['text'];found={x.lower()for x in re.findall(r'\b(thousands|millions|billions)\b',literal,re.I)}
  require(found=={name},'Conflicting or missing original unit literal')
  pattern=r'(?:\bin\s+|\(\s*)(?:thousands|millions|billions)\b|\b(?:thousands|millions|billions)\s*,?\s*except\b'
  require(re.search(pattern,literal,re.I),'Uniform formal-table unit declaration required')
 require(unique and bridge_cell.get('equal')is True,'Generic units require exact GAAP equality, not rounding compatibility')
 literal=cell.get('displayed_value');require(type(literal)is str and re.fullmatch(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?',literal),'Exact original displayed string required')
 value=positive(cell['value_decimal']);normalized=positive(cell['value_in_shares']);gaap=positive(bridge_cell['companyfacts_value_in_shares'])
 with localcontext()as ctx:
  ctx.prec=max(60,len(value.as_tuple().digits)+30,len(normalized.as_tuple().digits)+30,len(gaap.as_tuple().digits)+30)
  require(Decimal(literal.replace(',',''))==value and value*scale==normalized==gaap,'Exact original normalization/GAAP value mismatch')
 require(bridge_cell.get('original_value_in_shares')==cell['value_in_shares'] and bridge_cell.get('original_displayed_value')==literal and bridge_cell.get('original_share_scale')==str(scale),'Bridge original value strings mismatch')
 require(bridge_cell.get('original_source_sha256')==cell.get('source_sha256')==raw['source']['sha256'],'Bridge original source binding mismatch')
 require(bridge_cell.get('source_period_end')==cell['period_end']==selected_context['end'],'Bridge annual cell period mismatch')
 pointer=selected_context.get('json_pointer');require(type(pointer)is str and re.fullmatch(r'/facts/us-gaap/WeightedAverageNumberOfSharesOutstandingBasic/units/shares/\d+',pointer)and bridge_cell.get('json_pointer')==pointer,'Selected original GAAP Basic context mismatch')


def event_overlay(raw,avail,meta,bridge,clock):
 require(len({x['event_id'] for x in (raw,avail,meta,bridge)})==1,'Event axis mismatch')
 require(raw['symbol']==avail['symbol']==meta['symbol']==bridge['symbol'],'Symbol mismatch')
 require(raw['periods']==avail['periods']==meta['periods'],'Original periods changed')
 require(raw['inherited_holds']==avail['inherited_holds'],'Original holds changed')
 require(raw['identity']==avail['identity'],'Original identity changed')
 reasons=['INHERITED:'+x for x in raw['inherited_holds']['inherited_rejection_codes']]
 pair=raw['periods'];p,c=pair['comparison'],pair['current']
 normal=period_ok(p) and period_ok(c)
 if normal:normal=date.fromisoformat(c['period_start'])==date.fromisoformat(p['period_end'])+timedelta(days=1)
 if not normal:reasons.append('ORIGINAL_NORMAL_CONSECUTIVE_PERIODS_NOT_ESTABLISHED')
 unique=meta['status']==bridge['metadata_status']=='unique_same_accession_annual_pair'
 if not unique:reasons.append('UNIQUE_EARLIEST_CACHED_GAAP_PAIR_UNAVAILABLE')
 if unique:
  require(meta['selected_accession']==bridge['selected_accession'] and meta['selected_filed']==bridge['selected_filed'],'Bridge accession changed')
  for original_side,meta_side in (('current','current_selected'),('comparison','prior_selected')):
   context=meta[meta_side]
   require(context['accn']==meta['selected_accession'] and context['filed']==meta['selected_filed'],'Different accession context')
   pp=pair[original_side]
   require(context['end']==pp['period_end'],'Bridge period end mismatch')
   if pp['period_start'] is not None:require(context['start']==pp['period_start'],'Bridge period start mismatch')
 compatible=unique and all(bridge['cells'][side]['equal'] is True or
  bridge['cells'][side]['native_rounding_interval_compatible'] is True for side in ('current','comparison'))
 if not compatible:reasons.append('ORIGINAL_GAAP_DENOMINATOR_NOT_CORROBORATED')
 counts={}
 for side in ('current','comparison'):
  cell=raw['cells'][side];positive(cell['value_in_shares'])
  validate_unit(cell,bridge.get('cells',{}).get(side,{}),meta.get('current_selected'if side=='current'else'prior_selected',{}),raw,unique)
  require(cell['semantic_basis'] in ('basic_eps_denominator_weighting_implicit','explicit_basic_weighted_average_shares'), 'Wrong BASIC semantic basis')
  require(cell['period_end']==pair[side]['period_end'],'Cell not original annual period')
  counts[side]=cell['value_in_shares']
 known=clock(raw['release_date']);require(known==avail['known_at'],'Original knowledge clock changed')
 usable=max((known,clock(meta['selected_filed'])),key=stamp) if meta['selected_filed'] else known
 return dict(event_id=raw['event_id'],symbol=raw['symbol'],fiscal_year=raw['fiscal_year'],
  period_end=c['period_end'],prior_period_end=p['period_end'],known_at=known,usable_at=usable,
  current_shares=counts['current'],prior_shares=counts['comparison'],identity=deepcopy(raw['identity']),
  conditional_source_ready=not reasons,same_report_annual_pair=normal,same_share_basis=compatible,
  basis_is_declared_presentation_assumption=True,unresolved_reasons=reasons,
  inherited_holds=deepcopy(raw['inherited_holds']),original_event=deepcopy(raw),
  gaap_bridge=deepcopy(bridge),selected_gaap_contexts=deepcopy(meta),
  strict_source_admitted=False,old_admission_flags_changed=False)


def code_pins():return {SELF:sha(SELF),TEST:sha(TEST)}


def freeze():
 verify(OLD_PINS);v1=read(V1_PLAN);verify(v1['input_hashes']);verify(v1['implementation_hashes'])
 require(read(POLICY)==policy(),'Only uniform-unit counts policy may change')
 require(sha(old.OUT+'/TESTS.json')==v1['tests_sha256'],'Frozen v1 tests changed')
 require(read(FAILURE)['exit_code']==1,'Prior failure evidence required')
 tests=read(OUT+'/TESTS.json');require(tests['passed']is True and tests['synthetic_only']is True and tests['implementation_hashes']==code_pins(),'Current passing v2 tests required')
 pins=merge(v1['input_hashes'],v1['implementation_hashes'],OLD_PINS,{POLICY:sha(POLICY),FAILURE:sha(FAILURE),old.OUT+'/TESTS.json':sha(old.OUT+'/TESTS.json')})
 verify(pins)
 write(OUT+'/PLAN.json',dict(schema='basic_shares_conditional_admission_plan_v2',created_utc=datetime.now(timezone.utc).isoformat(),policy=read(POLICY),input_hashes=pins,implementation_hashes=code_pins(),tests_sha256=sha(OUT+'/TESTS.json'),supersedes_failed_plan={'path':V1_PLAN,'sha256':OLD_PINS[V1_PLAN]},change='Only fixed generic formal-table unit guard; original inputs and all other event/row rules preserved.',ratios_computed=0,real_strategy_trials=0))
 print({'plan_sha256':sha(OUT+'/PLAN.json')})


def run(expected):
 require(sha(OUT+'/PLAN.json')==expected,'Explicit v2 PLAN required');plan=read(OUT+'/PLAN.json');verify(plan['input_hashes']);verify(plan['implementation_hashes']);require(sha(OUT+'/TESTS.json')==plan['tests_sha256'],'V2 tests changed');require(plan['policy']==policy(),'V2 policy changed')
 import after158_basic_shares_availability as av
 sessions,closes=av.calendar();clock=lambda d:av.known_clock(d,sessions,closes)['iso']
 raw=read(old.RAW)['events'];ae=read(old.AV+'/EVENTS.json')['events'];bm=read(old.BR+'/METADATA.json')['records'];br=read(old.BR+'/COMPARISON.json')['records']
 require(len(raw)==len(ae)==len(bm)==len(br)==44,'44 original events required')
 generic=[e for e in raw if e['cells']['current']['unit']['share_specific']is False]
 require(len(generic)==34 and all(e['symbol']in GENERIC_YEARS and e['fiscal_year']in GENERIC_YEARS[e['symbol']]for e in generic),'Exact original 34-event generic scope')
 events=[event_overlay(*parts,clock)for parts in zip(raw,ae,bm,br)]
 for e,original in zip(events,raw):require(e['original_event']==original and e['inherited_holds']==original['inherited_holds'],'Original values, flags or holds mutated')
 original=read(old.AV+'/ROWS.json');require(len(original)==5978,'Full5978 rows required');rows=[map_row(row,events)for row in original];s=summary(events,rows)
 verify(plan['input_hashes']);verify(plan['implementation_hashes'])
 write(OUT+'/EVENTS.json',{'events':events,'strict_source_admitted':False});write(OUT+'/ROWS.json',rows);write(OUT+'/COVERAGE.json',s)
 outputs={OUT+'/'+n:sha(OUT+'/'+n)for n in('EVENTS.json','ROWS.json','COVERAGE.json')}
 write(OUT+'/MANIFEST.json',dict(schema='basic_shares_conditional_admission_manifest_v2',plan_sha256=expected,input_hashes=merge(plan['input_hashes'],plan['implementation_hashes'],{OUT+'/PLAN.json':expected,OUT+'/TESTS.json':sha(OUT+'/TESTS.json')}),outputs=outputs,summary=s,strict_source_admitted=False,old_admission_flags_changed=False,numerical_experiment_registered=False,activity={'ratios':0,'ranks':0,'scores':0,'orders':0,'returns':0},limitations=plan['policy']['source_limitations']))
 print(old.json.dumps({'summary':s,'manifest_sha256':sha(OUT+'/MANIFEST.json')},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--freeze',action='store_true');g.add_argument('--plan-sha');a=p.parse_args()
 if a.prepare:prepare()
 elif a.freeze:freeze()
 else:run(a.plan_sha)
