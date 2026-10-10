"""Profiling is separate from official timing; extract actual module shapes."""
import torch
from common.reporting import write_json


def collect_profile(adapter, operation, directory, backend):
    if backend.device.type == 'musa':
        raise RuntimeError('MUSA native profiler must be verified against installed torch_musa/torchada version; CUDA activity is not an assumed substitute')
    shapes, handles = {}, []
    def capture(name):
        def hook(module, inputs):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            spec = {"module": name, "type": type(module).__name__, "input_shape": list(x.shape),
                    "dtype": str(x.dtype), "stride": list(x.stride())}
            if isinstance(module, torch.nn.Linear):
                spec["weight_shape"] = list(module.weight.shape)
            if isinstance(module, torch.nn.LayerNorm):
                spec["normalized_shape"] = list(module.normalized_shape)
            if hasattr(module, "n_heads"):
                spec["n_heads"] = module.n_heads
                seq_id = inputs[1] if len(inputs) > 1 else None
                spec["attention_mask"] = "None" if seq_id is None else "pairwise equality of sequence_id"
                spec["sequence_id_shape"] = None if seq_id is None else list(seq_id.shape)
            signature = repr(spec)
            if signature in shapes:
                shapes[signature]['invocations'] += 1
            else:
                shapes[signature] = dict(spec, invocations=1)
        return hook
    for name, module in adapter.model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) or hasattr(module, "n_heads"):
            handles.append(module.register_forward_pre_hook(capture(name)))
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if backend.device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,
                                    record_shapes=True, profile_memory=True) as profile:
            with torch.profiler.record_function("esm3_task"):
                result = operation()
            backend.synchronize()
        del result
    finally:
        for handle in handles:
            handle.remove()
    profile.export_chrome_trace(str(directory / "trace.json"))
    rows = []
    for item in profile.key_averages(group_by_input_shape=True):
        rows.append({"name": item.key, "count": item.count, "input_shapes": item.input_shapes,
                     "device_type": str(item.device_type), "is_aten_operator": item.key.startswith("aten::"),
                     "self_cuda_us": item.self_cuda_time_total, "total_cuda_us": item.cuda_time_total,
                     "self_cpu_us": item.self_cpu_time_total})
    rows.sort(key=lambda row: row["self_cuda_us"], reverse=True)
    write_json(directory / "operators.json", {"scope": "single separate profiled call; filter ATen operators for operator aggregation, do not add kernel or user-range rows; self time excludes children; profiling excluded from benchmark",
                                             "operators": rows})
    write_json(directory / "operator_shapes.json", list(shapes.values()))
