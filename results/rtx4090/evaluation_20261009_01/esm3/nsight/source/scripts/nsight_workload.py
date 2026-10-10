"""Native CUDA capture with explicit task, batch, precision and pipeline scope."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('HF_HUB_OFFLINE', '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from common.backend import Backend
from common.reporting import environment, sha256, write_json
from main import load_samples
from models.esm3_adapter import ESM3Adapter


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'config': vars(args), 'scope': args.scope, 'benchmark': False,
              'scope_definition': 'warm: prepared operation only; pipeline: model load + prepare + operation + CPU artifacts/PDB, excluding .pt serialization'}
    phases = []
    try:
        torch.set_num_threads(args.cpu_threads)
        backend = Backend(args.device)
        if backend.device.type != 'cuda':
            raise ValueError('Nsight is CUDA-only; use a validated MUSA profiling tool on S5000')
        backend.seed(args.seed)
        samples = load_samples(args.inputs, args.sample_ids or [args.sample_id], args.batch_size)
        @contextmanager
        def phase(name):
            backend.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.nvtx.range('esm3_phase_' + name):
                yield
                backend.synchronize()
            phases.append({'name': name, 'instrumented_host_seconds': time.perf_counter() - t0})
        def prepare_model():
            with phase('model_load'):
                adapter = ESM3Adapter(backend, args.precision, args.manifest, args.attention)
            with phase('prepare'):
                prepared = adapter.prepare(args.task, samples, args.pdb, args.mask_positions, args.steps)
            return adapter, prepared
        with torch.inference_mode(), backend.context():
            if args.scope == 'warm':
                adapter, prepared = prepare_model()
                for _ in range(args.warmup):
                    backend.seed(args.seed)
                    result = prepared['operation']()
                    del result
                backend.synchronize()
                phases.clear()
            if args.tool == 'nsys':
                torch.cuda.profiler.start()
            try:
                with torch.cuda.nvtx.range('esm3_capture'):
                    if args.scope == 'pipeline':
                        adapter, prepared = prepare_model()
                    for i in range(args.calls):
                        backend.seed(args.seed)
                        with phase(f'{args.task}_{i}'):
                            result = prepared['operation']()
                        if i + 1 < args.calls:
                            del result
                    if args.scope == 'pipeline':
                        with phase('deliver_cpu_artifacts'):
                            payload, quality = adapter.artifacts(prepared, result, output)
            finally:
                if args.tool == 'nsys':
                    torch.cuda.profiler.stop()
            if args.scope == 'warm':
                payload, quality = adapter.artifacts(prepared, result, output)
            torch.save(payload, output / 'tensors.pt')
        exact = None
        if args.reference:
            reference = torch.load(args.reference, map_location='cpu', weights_only=True)
            exact = {key: torch.equal(reference[key], value) for key, value in payload.items()
                     if isinstance(value, torch.Tensor) and key in reference and args.task == 'forward'}
            if args.task != 'forward':
                exact = None
        record.update(environment=environment(REPO), device=backend.metadata(), quality=quality,
                      weights={'revision': adapter.manifest['revision'], 'manifest_sha256': sha256(args.manifest)},
                      precision=adapter.precision_metadata(), memory=backend.memory(), phases=phases,
                      reference=args.reference, full_tensor_exact=exact,
                      validation_rule='functional task checks; optional exact forward comparison on the same configuration and hardware',
                      residue_lengths=[len(s['sequence']) for s in samples], input_sha256=sha256(args.inputs),
                      status='pass' if exact is None or (bool(exact) and all(exact.values())) else 'fail')
    except Exception as exc:
        record.update(status='error', error=str(exc), traceback=traceback.format_exc(), phases=phases)
        write_json(output / 'workload.json', record)
        raise
    write_json(output / 'workload.json', record)
    if record['status'] != 'pass':
        raise RuntimeError('Captured forward differs from supplied reference')
    print(f'PROFILE_WORKLOAD_PASS {args.task} {args.scope}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-id', default='ubiquitin_human')
    p.add_argument('--sample-ids', nargs='+')
    p.add_argument('--inputs', default=str(REPO / 'inputs/test_sequences.jsonl'))
    p.add_argument('--manifest', default=str(REPO.parent / 'esm-baseline/model_snapshot.json'))
    p.add_argument('--pdb', default=str(REPO.parent / 'esm-baseline/data/1UBQ.pdb'))
    p.add_argument('--reference')
    p.add_argument('--output', required=True)
    p.add_argument('--tool', choices=['nsys', 'ncu'], required=True)
    p.add_argument('--scope', choices=['warm', 'pipeline'], default='warm')
    p.add_argument('--task', choices=['forward', 'sequence', 'structure', 'inverse'], default='forward')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--attention', choices=['auto', 'math'], default='auto')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--mask-positions', nargs='+', type=int, default=[4, 5, 13])
    p.add_argument('--steps', type=int)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--calls', type=int, default=5)
    a = p.parse_args()
    if min(a.calls, a.batch_size, a.cpu_threads) < 1 or a.warmup < 0:
        p.error('calls/batch/threads must be positive; warmup nonnegative')
    if a.scope == 'pipeline' and a.calls != 1:
        p.error('pipeline scope requires calls=1')
    run(a)
