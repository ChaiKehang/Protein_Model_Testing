"""Native Systems workload: full prepared OpenFold model; no formal benchmark claim."""
import argparse
from pathlib import Path
import sys
from contextlib import contextmanager
import time
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from common.backend import Backend
from common.reporting import sha256,write_json
from models.openfold_adapter import OpenFoldAdapter


def main(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    reference=Path(a.reference).resolve();report=__import__('json').loads((reference/'report.json').read_text())
    if report['status']!='pass':raise ValueError('Reference benchmark did not pass')
    backend=Backend(a.device)
    if backend.device.type!='cuda':raise ValueError('NVIDIA capture requires actual CUDA')
    cfg=report['parameters'];torch.set_num_threads(cfg['cpu_threads'])
    if sha256(cfg['weights'])!=report['weights']['sha256']:raise ValueError('Reference weights differ')
    phases=[]
    @contextmanager
    def phase(name):
        backend.synchronize();t=time.perf_counter()
        with torch.cuda.nvtx.range('openfold_phase_'+name):
            yield
            backend.synchronize()
        phases.append({'name':name,'instrumented_seconds':time.perf_counter()-t})
    def load():
        with phase('model_load'):
            adapter=OpenFoldAdapter(backend,cfg['weights'],cfg['chunk_size'],cfg['seed'],cfg.get('attention_kernel','upstream'))
        with phase('prepare'):
            adapter.prepare(cfg['fasta'],cfg['alignments'],reference/'features.pt' if a.scope=='warm' else None)
        if adapter.feature_sha256!=report['features']['feature_sha256']:raise ValueError('Reference features differ')
        return adapter
    if a.scope=='warm':
        adapter=load()
        for _ in range(a.warmup):
            result=adapter.run();backend.synchronize();del result
        phases.clear()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        with torch.cuda.nvtx.range('openfold_capture'):
            if a.scope=='pipeline':adapter=load()
            with phase('task_0'):
                result=adapter.run();backend.synchronize()
            if a.scope=='pipeline':
                with phase('deliver'):
                    (root/'prediction_unrelaxed.pdb').write_text(adapter.pdb(result))
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    pred=result['final_atom_positions'].cpu()
    saved=torch.load(reference/'tensors.pt',map_location='cpu',weights_only=True)
    mask=saved['final_atom_mask'].bool()
    exact=torch.equal(pred[mask],saved['final_atom_positions'][mask])
    maximum=float((pred[mask]-saved['final_atom_positions'][mask]).abs().max())
    write_json(root/'workload.json',{'status':'pass' if exact else 'fail','reference_report_sha256':sha256(reference/'report.json'),
        'feature_sha256':adapter.feature_sha256,'capture':a.scope,'phases':phases,
        'scope_note':'warm: prepared model; pipeline: load, actual MSA feature preparation, prediction and CPU/PDB delivery; excludes MSA search/templates/relaxation',
        'exact_valid_positions':exact,'max_abs_coordinate_difference':maximum,
        'source_sha256':{n:sha256(REPO/n) for n in ['scripts/openfold_nsight_workload.py','models/openfold_adapter.py']}})
    if not exact:raise ValueError('Profile replay coordinate differs from frozen reference')
    print('PASS: native OpenFold capture')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--scope',choices=['warm','pipeline'],default='warm')
    p.add_argument('--warmup',type=int,default=1);a=p.parse_args()
    if a.warmup<1:p.error('Warmup must be positive')
    main(a)
