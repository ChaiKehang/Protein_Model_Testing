"""Optional NVIDIA telemetry keyed to logical CUDA device identity, never physical GPU 0."""
import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager


def cuda_selector(device):
    if not device.startswith('cuda'):
        return {'status':'unsupported_backend', 'reason':'NVIDIA telemetry is CUDA-only; MUSA adapter not verified'}
    try:
        library=ctypes.CDLL('libcuda.so.1')
        library.cuInit.argtypes=[ctypes.c_uint];library.cuInit.restype=ctypes.c_int
        library.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int];library.cuDeviceGet.restype=ctypes.c_int
        device_handle=ctypes.c_int()
        ordinal=int(device.split(':')[1]) if ':' in device else 0
        for name,code in [('cuInit',library.cuInit(0)),('cuDeviceGet',library.cuDeviceGet(ctypes.byref(device_handle),ordinal))]:
            if code:raise RuntimeError(f'{name} failed: {code}')
        class CUuuid(ctypes.Structure):
            _fields_=[('bytes',ctypes.c_ubyte*16)]
        value=CUuuid();fn=getattr(library,'cuDeviceGetUuid_v2',library.cuDeviceGetUuid)
        fn.argtypes=[ctypes.POINTER(CUuuid),ctypes.c_int];fn.restype=ctypes.c_int
        code=fn(ctypes.byref(value),device_handle)
        if code:raise RuntimeError(f'cuDeviceGetUuid failed: {code}')
        return {'status':'pass','selector':'GPU-'+str(uuid.UUID(bytes=bytes(value.bytes))),
                'logical_device':device,'mapping':'CUDA driver logical ordinal -> device UUID, respecting visibility'}
    except Exception as exc:
        return {'status':'unavailable','reason':str(exc)}


def start_monitor(device, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    query=subprocess.run([sys.executable,__file__,'--device',device],capture_output=True,text=True)
    try:info=json.loads(query.stdout)
    except ValueError:info={'status':'unavailable','reason':query.stdout+query.stderr}
    executable=shutil.which('nvidia-smi')
    handles=[];process=None
    if info['status']=='pass' and executable:
        probe=subprocess.run([executable,'-i',info['selector'],'--query-gpu=uuid,name','--format=csv'],capture_output=True,text=True)
        if probe.returncode==0:
            output=(directory/'gpu_samples.csv').open('w');error=(directory/'gpu_monitor.log').open('w');handles=[output,error]
            command=[executable,'--query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
                     '--format=csv','-lms','500','-i',info['selector']]
            process=subprocess.Popen(command,stdout=output,stderr=error)
            info.update(status='collecting',command=command,probe=probe.stdout)
        else:info.update(status='unavailable',reason=probe.stderr)
    elif not executable and info['status']=='pass':info.update(status='unavailable',reason='nvidia-smi not installed')
    (directory/'telemetry_capability.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    return process,handles


def stop_monitor(process, handles):
    if process:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    for handle in handles:handle.close()


@contextmanager
def monitoring(device, directory):
    process,handles=start_monitor(device,directory)
    try:yield
    finally:stop_monitor(process,handles)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True)
    print(json.dumps(cuda_selector(parser.parse_args().device)))
