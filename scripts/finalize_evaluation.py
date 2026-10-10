"""Verify the delivery and derive observed operator coverage from measured files."""
import argparse
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote,urlsplit
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from common.reporting import write_json,sha256


def main(a):
 root=Path(a.output).resolve()
 write_json(root/'verification.json',{'status':'running'})
 observed={}
 for p in list((root/'esm3').glob('*/operators.json'))+[root/'openfold/operators.json']:
  data=json.loads(p.read_text());observed[str(p.relative_to(root))]=sorted({r['name'] for r in data['operators'] if r['name'].startswith('aten::')})
 family_patterns={'GEMM':['aten::mm','aten::linear','aten::addmm'],'BatchGEMM':['aten::bmm'],'Attention':['scaled_dot_product','flash_attention'],
                  'Softmax':['softmax'],'LayerNorm':['layer_norm'],'RMSNorm':['rms_norm'],'Embedding':['embedding'],
                  'Convolution':['conv','convolution'],'Reduction':['aten::sum','aten::mean','norm'],
                  'Gather/Scatter':['gather','scatter','index_select'],'TopK':['topk'],'Diffusion':[],'Graph':[]}
 rows=[]
 for family,patterns in family_patterns.items():
  match={p:[n for n in names if any(part in n for part in patterns)] for p,names in observed.items()};match={p:v for p,v in match.items() if v}
  rows.append({'family':family,'status':'observed_ATen' if match else 'not_observed_in_profiled_configuration','evidence':match,
               'note':'Unobserved is not unsupported; name matching does not establish composite model semantics'})
 modules=json.loads((root/'openfold/protein_modules.json').read_text())
 for family,term in [('TriangleAttention','TriangleAttention'),('TriangleMultiplication','TriangleMultiplication'),('OuterProductMean','OuterProduct'),('IPA','InvariantPoint')]:
  found=[r for r in modules if term in r['type']]
  rows.append({'family':family,'status':'observed_model_modules' if found else 'not_observed','evidence':found})
 geometry=[]
 for p in (root/'esm3').glob('*/operator_shapes.json'):
  geometry.extend({'case':p.parent.name,**r} for r in json.loads(p.read_text()) if '.geom_attn.' in r['module'])
 rows.append({'family':'3DGeometry','status':'observed_model_submodules' if geometry else 'not_observed','evidence':geometry})
 write_json(root/'operator_coverage.json',{'families':rows,'scope':'Observed original model calls only; no MUSA kernel or unused branch claims'})
 metric_rows=[];artifact_checks=[];core_checks=[]
 for path in list((root/'esm3').glob('*/report.json'))+list((root/'esm3_standard').glob('*/report.json')):
  r=json.loads(path.read_text())
  if r['status']!='pass':continue
  metric_rows.append({'model':'ESM3','case':path.parent.name,'task':r['task'],'precision':r['config']['precision'],
       'batch':r['config']['batch_size'],'lengths':str(r['residue_lengths']),'median_seconds':r['latency']['median'],
       'p90_seconds':r['latency']['p90'],'cv_percent':r['latency']['cv_percent'],
       'proteins_per_second':r['throughput']['proteins_per_second'],'allocated_peak_gib':r['memory']['allocated_peak_gib']})
  for name in ['models/esm3_adapter.py','common/backend.py','common/reporting.py','main.py']:
   core_checks.append({'report':str(path),'source':name,'matches_current':r['environment']['source_sha256'][name]==sha256(REPO/name)})
 of=json.loads((root/'openfold/report.json').read_text());lat=of['latency_seconds']
 metric_rows.append({'model':'OpenFold','case':'6KWC','task':'structure','precision':'fp32','batch':1,'lengths':str([of['output']['residues']]),
       'median_seconds':lat['median'],'p90_seconds':lat['p90'],'cv_percent':lat['cv_percent'],
       'proteins_per_second':len(of['samples_seconds'])/sum(of['samples_seconds']),'allocated_peak_gib':of['memory']['allocated_peak_gib']})
 for name,digest in of['artifacts_sha256'].items():
  artifact_checks.append({'artifact':name,'sha256_matches':sha256(root/'openfold'/name)==digest})
 with (root/'metrics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.targets=[]
  def handle_starttag(self,tag,attrs):
   for name,value in attrs:
    if name in ['href','src'] and value:self.targets.append(value)
 parser=Links();parser.feed((root/'report.html').read_text());local=[];missing=[]
 for value in parser.targets:
  if value.startswith(('#','data:')) or urlsplit(value).scheme:continue
  target=(root/unquote(value.split('#',1)[0])).resolve();local.append(str(target))
  if not target.exists():missing.append(str(target))
 provenance=json.loads((root/'report_provenance.json').read_text());mismatches=[]
 for path,digest in provenance['sources'].items():
  if not Path(path).is_file() or sha256(path)!=digest:mismatches.append(path)
 if sha256(root/'report.html')!=provenance['html_sha256']:mismatches.append('report.html')
 for name,digest in provenance['figures'].items():
  if sha256(root/'figures'/name)!=digest:mismatches.append('figures/'+name)
 native=[]
 s=json.loads((root/'esm3/nsight/nsight_summary.json').read_text())
 for name,r in s['systems'].items():
  p=root/'esm3/nsight'/r['native_report'];native.append({'path':str(p),'sha256_matches':sha256(p)==r['sha256'],'status':r['status']})
 extra=root/'esm3_standard/nsight/nsight_summary.json'
 if extra.exists():
  for name,r in json.loads(extra.read_text())['systems'].items():
   p=extra.parent/r['native_report'];native.append({'path':str(p),'sha256_matches':sha256(p)==r['sha256'],'status':r['status']})
 for folder in ['openfold_nsight','openfold_pipeline']:
  r=json.loads((root/folder/'systems_summary.json').read_text());native.append({'path':r['native_report'],'sha256_matches':sha256(r['native_report'])==r['native_sha256'],'status':r['status']})
 tests=Path(a.tests).resolve()
 catalog=json.loads((root/'experiment_index.json').read_text())
 records=catalog['records']; ids=[r['id'] for r in records]
 catalog_errors=[]
 if len(ids)!=len(set(ids)):catalog_errors.append('Duplicate experiment IDs')
 expected_cases={str(p.parent.relative_to(root)) for p in
                 list((root/'esm3').glob('*/report.json'))+list((root/'esm3_standard').glob('*/report.json'))}
 indexed_cases={r['id'] for r in records if r['group'] in ('core','standard')}
 if indexed_cases!=expected_cases:catalog_errors.append('Formal case coverage differs from saved reports')
 for r in records:
  for path in r['evidence']:
   if not Path(path).is_file():catalog_errors.append(r['id']+': missing evidence '+path)
  if r['group'] in ('core','standard'):
   raw=json.loads(Path(r['source']).read_text());cfg=raw['config']
   actual=(raw['task'],cfg['batch_size'],raw['residue_lengths'],cfg['precision'],cfg['device'],cfg['warmup'],raw['latency']['count'])
   indexed=(r['task'],r['batch_size'],r['residue_lengths'],r['precision'],r['device'],r['warmup'],r['runs'])
   if indexed!=actual:catalog_errors.append(r['id']+': settings differ from actual report')
  if r['group']=='fitness':
   sizes=r['settings']['context_batch_sizes']
   if sum(sizes)!=r['settings']['masked_contexts'] or any(b>r['batch_size'] or b<1 for b in sizes):
    catalog_errors.append(r['id']+': invalid actual context batches')
  generation=r['settings'].get('generation')
  if generation and generation.get('source') and sha256(generation['source'])!=generation['source_sha256']:
   catalog_errors.append(r['id']+': generation source hash differs')
 indexed_native={str(Path(r['evidence'][-1]).resolve()) for r in records if r['group']=='systems'}
 if indexed_native!={str(Path(r['path']).resolve()) for r in native}:catalog_errors.append('Native report coverage differs')
 result={'status':'pass' if not missing and not mismatches and all(r['sha256_matches'] and r['status']=='pass' for r in native)
         and all(r['matches_current'] for r in core_checks) and all(r['sha256_matches'] for r in artifact_checks) and not catalog_errors else 'fail',
         'html_local_links':len(local),'missing_local_files':missing,'source_or_figure_hash_mismatches':mismatches,'native_reports':native,
         'unit_test_log':str(tests),'unit_test_sha256':sha256(tests),'browser_render_test':None,
         'inference_source_checks':core_checks,'openfold_artifact_checks':artifact_checks,
         'experiment_index_checks':{'experiment_count':len(records),'formal_case_count':len(indexed_cases),
                                    'native_report_count':len(indexed_native),'errors':catalog_errors},
         'browser_note':'HTML structure, links, source hashes and exported figures checked; no browser screenshot claimed',
         'evaluation_failures_preserved':'BF16 batch checks may remain partial; verification PASS is delivery integrity, not all numerical gates',
         'current_scope':'4090 reference evaluation only; no optimization or future-hardware acceptance'}
 write_json(root/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
 if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--tests',required=True);main(p.parse_args())
