"""Read native Nsight exports. Units, interval unions and denominators are explicit."""
import csv
import io
import math
from pathlib import Path
import sqlite3
import statistics


def interval_union_ns(intervals, start, end):
    clipped=sorted((max(a,start),min(b,end)) for a,b in intervals if b>start and a<end and b>a)
    total=0; previous=None
    for a,b in clipped:
        if previous is None:previous=[a,b]
        elif a<=previous[1]:previous[1]=max(previous[1],b)
        else:total+=previous[1]-previous[0];previous=[a,b]
    return total+(previous[1]-previous[0] if previous else 0)


def read_systems(path, capture_name='esm3_capture'):
    phase_prefixes = (capture_name.removesuffix('_capture') + '_phase_',)
    if capture_name == 'esm3_capture':
        phase_prefixes += ('esm3_forward_',)
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as db:
        db.row_factory=sqlite3.Row
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        required={'StringIds','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_KERNEL'}
        if not required<=tables:raise ValueError(f'Missing required Nsight tables: {required-tables}')
        strings=dict(db.execute('select id,value from StringIds'))
        ranges=[]
        for row in db.execute('select start,end,text,textId from NVTX_EVENTS where end is not null'):
            text=row['text'] or strings.get(row['textId'],'')
            if text==capture_name or text.startswith(phase_prefixes):
                ranges.append({'name':text,'start_ns':row['start'],'end_ns':row['end']})
        outer=[r for r in ranges if r['name']==capture_name]
        if len(outer)!=1:raise ValueError(f'Expected exactly one completed {capture_name} range')
        start,end=outer[0]['start_ns'],outer[0]['end_ns']
        if end<=start:raise ValueError('Invalid capture interval')
        kernels=[];grouped={}
        for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?',(start,end)):
            name=strings[row['demangledName']]
            a,b=max(start,row['start']),min(end,row['end'])
            entry={'name':name,'start_ns':a,'end_ns':b,'duration_ns':b-a,'stream':row['streamId'],
                   'grid':[row[f'grid{axis}'] for axis in 'XYZ'],'block':[row[f'block{axis}'] for axis in 'XYZ'],
                   'registers_per_thread':row['registersPerThread'],
                   'device_id':row['deviceId'] if 'deviceId' in row.keys() else None,
                   'static_shared_bytes':row['staticSharedMemory'] if 'staticSharedMemory' in row.keys() else None,
                   'dynamic_shared_bytes':row['dynamicSharedMemory'] if 'dynamicSharedMemory' in row.keys() else None}
            kernels.append(entry);grouped.setdefault(name,[]).append(entry)
        if not kernels:raise ValueError('No CUDA kernels collected in the capture range')
        devices={k['device_id'] for k in kernels if k['device_id'] is not None}
        if len(devices)>1:raise ValueError('Multiple CUDA devices require per-device activity denominators; single-device parser refuses aggregation')
        total=sum(k['duration_ns'] for k in kernels)
        summaries=[]
        for name,items in grouped.items():
            durations=[k['duration_ns'] for k in items]
            summaries.append({'name':name,'instances':len(items),'total_ms':sum(durations)/1e6,'mean_us':statistics.mean(durations)/1e3,
                              'median_us':statistics.median(durations)/1e3,'kernel_sum_percent':sum(durations)/total*100,
                              'first_grid':items[0]['grid'],'first_block':items[0]['block']})
        summaries.sort(key=lambda r:r['total_ms'],reverse=True)
        variants = {}
        for entry in kernels:
            signature = (entry['name'], tuple(entry['grid']), tuple(entry['block']), entry['registers_per_thread'],
                         entry['static_shared_bytes'],entry['dynamic_shared_bytes'])
            variants.setdefault(signature, []).append(entry)
        variant_rows = []
        for signature, items in variants.items():
            name, grid, block, registers, static_shared, dynamic_shared = signature
            variant_rows.append({'name': name, 'grid': list(grid), 'block': list(block),
                                 'registers_per_thread': registers, 'instances': len(items),
                                 'static_shared_bytes':static_shared,'dynamic_shared_bytes':dynamic_shared,
                                 'total_ms': sum(i['duration_ns'] for i in items) / 1e6,
                                 'kernel_sum_percent': sum(i['duration_ns'] for i in items) / total * 100,
                                 'first_start_ns': min(i['start_ns'] for i in items)})
        variant_rows.sort(key=lambda r: r['total_ms'], reverse=True)
        transfers=[]
        kinds={}
        if 'ENUM_CUDA_MEMCPY_OPER' in tables:kinds={row['id']:row['label'] for row in db.execute('select * from ENUM_CUDA_MEMCPY_OPER')}
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            for row in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?',(start,end)):
                transfers.append({'kind':kinds.get(row['copyKind'],str(row['copyKind'])),'bytes':row['bytes'],
                                  'start_ns':max(start,row['start']),'end_ns':min(end,row['end'])})
        memsets=[]
        if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
            memsets=[(max(start,r[0]),min(end,r[1])) for r in db.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMSET where end>? and start<?',(start,end))]
        api_groups={}
        if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
            for row in db.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where end>? and start<?',(start,end)):
                name=strings[row['nameId']];api_groups.setdefault(name,[]).append(min(end,row['end'])-max(start,row['start']))
        apis=sorted([{'name':n,'calls':len(v),'total_ms':sum(v)/1e6} for n,v in api_groups.items()],key=lambda r:r['total_ms'],reverse=True)
        kernel_intervals=[(k['start_ns'],k['end_ns']) for k in kernels]
        activity_intervals=kernel_intervals+[(r['start_ns'],r['end_ns']) for r in transfers]+memsets
        kernel_active=interval_union_ns(kernel_intervals,start,end)
        active=interval_union_ns(activity_intervals,start,end)
        transfer_groups={}
        for r in transfers:
            value=transfer_groups.setdefault(r['kind'],{'kind':r['kind'],'calls':0,'bytes':0,'total_ms':0})
            value['calls']+=1;value['bytes']+=r['bytes'];value['total_ms']+=(r['end_ns']-r['start_ns'])/1e6
        iterations=[]
        for r in ranges:
            if r['name'].startswith(phase_prefixes):
                a,b=r['start_ns'],r['end_ns'];busy=interval_union_ns(activity_intervals,a,b)
                iterations.append({'name':r['name'],'start_ns':a,'end_ns':b,'range_ms':(b-a)/1e6,'gpu_activity_ms':busy/1e6,'no_gpu_activity_ms':(b-a-busy)/1e6})
        return {'capture_name':capture_name,'captured_cuda_device_ids':sorted(devices),
                'capture_range_ms':(end-start)/1e6,'kernel_count':len(kernels),'kernel_sum_ms':total/1e6,
                'kernel_active_union_ms':kernel_active/1e6,'gpu_activity_union_ms':active/1e6,
                'no_gpu_activity_ms':(end-start-active)/1e6,'gpu_activity_fraction_percent':active/(end-start)*100,
                'kernels':summaries,'kernel_launch_variants':variant_rows,
                'cuda_apis':apis,'transfers':list(transfer_groups.values()),'iterations':iterations,
                'timeline_origin_ns':start,'timeline_kernels':kernels,
                'semantics':{'kernel_percent':'sum of durations for this name / sum of all kernel durations; not wall-time share',
                             'activity_fraction':'union of kernel, memcpy, memset intervals / captured host NVTX range; not SM occupancy or GPU utilization',
                             'gap':'no recorded GPU activity; does not identify a CPU root cause',
                             'timing':'instrumented capture; not replacement for saved benchmark'}}


def read_compute_csv(path):
    text=Path(path).read_text(encoding='utf-8')
    lines=text.splitlines()
    first=next((i for i,line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line),None)
    if first is None:raise ValueError('Nsight Compute raw CSV has no metric header; do not treat failed collection as zero')
    rows=list(csv.DictReader(io.StringIO('\n'.join(lines[first:]))))
    launches={row['ID'] for row in rows if row.get('ID') and row.get('Metric Name')}
    if len(launches)>1:
        raise ValueError('Multiple Compute launches require explicit per-launch grouping')
    metrics={};raw=[]
    for row in rows:
        name=row.get('Metric Name');value=row.get('Metric Value');unit=row.get('Metric Unit')
        if not name or value is None:continue
        raw.append(row)
        try:
            number=float(value.replace(',',''))
            if not math.isfinite(number):number=None
        except ValueError:number=None
        metrics[name]={'value':number,'unit':unit,'raw_value':value}
    if not metrics:raise ValueError('No metrics in Nsight Compute export')
    selected={}
    wanted={'duration':'gpu__time_duration.sum','compute_percent':'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'memory_percent':'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
            'dram_percent':'dram__throughput.avg.pct_of_peak_sustained_elapsed',
            'occupancy_percent':'sm__warps_active.avg.pct_of_peak_sustained_active',
            'dram_read_bytes':'dram__bytes_read.sum','dram_write_bytes':'dram__bytes_write.sum',
            'registers_per_thread':'launch__registers_per_thread','threads':'launch__thread_count'}
    for label,key in wanted.items():selected[label]=metrics.get(key)
    return {'metrics':metrics,'selected_metrics':selected,'raw_rows':raw,
            'semantics':'one selected kernel launch, default kernel replay; hardware counters are not whole-model averages'}


def compute_targets(analysis, top_k=3):
    """Plan representative launches by duration, preserving launch parameters.

    Ordinal is within the name-filtered first operation range, not all CUDA launches.
    Counter collection must still verify the native report's selected grid/block.
    """
    operations = [r for r in analysis['iterations'] if r['name'].startswith('esm3_forward_')
                  or r['name'].startswith(('esm3_phase_', 'openfold_phase_')) and r['name'].rsplit('_', 1)[-1].isdigit()]
    if not operations:
        raise ValueError('No operation range for representative launch selection')
    first = min(operations, key=lambda r:r['start_ns'])
    within = sorted([k for k in analysis['timeline_kernels'] if k['start_ns'] >= first['start_ns']
                     and k['end_ns'] <= first['end_ns']], key=lambda k:k['start_ns'])
    result = []
    for family in analysis['kernels'][:top_k]:
        same = [k for k in within if k['name'] == family['name']]
        if not same:
            result.append({'name':family['name'], 'status':'no_launch_in_first_operation'})
            continue
        groups = {}
        for ordinal, k in enumerate(same):
            sig = (tuple(k['grid']), tuple(k['block']), k['registers_per_thread'],
                   k.get('static_shared_bytes'),k.get('dynamic_shared_bytes'))
            groups.setdefault(sig, []).append((ordinal,k))
        signature, values = max(groups.items(), key=lambda pair:sum(k['duration_ns'] for _,k in pair[1]))
        result.append({'name':family['name'],'status':'planned', 'filtered_launch_skip':values[0][0],
                       'grid':list(signature[0]),'block':list(signature[1]),'registers_per_thread':signature[2],
                       'static_shared_bytes':signature[3],'dynamic_shared_bytes':signature[4],
                       'family_kernel_sum_percent':family['kernel_sum_percent'],
                       'selection':'highest aggregate-duration launch variant of this family in the first operation',
                       'nvtx_operation_range':first['name'], 'verification_required':'native NCU launch metadata must match; not yet counter evidence'})
    return result


def verify_compute_launch(metrics, selection):
    fields={f'launch__{kind}_dim_{axis}':selection[kind][i]
            for kind in ['grid','block'] for i,axis in enumerate('xyz')}
    fields['launch__registers_per_thread']=selection['registers_per_thread']
    missing=[key for key in fields if metrics.get(key,{}).get('value') is None]
    mismatched={key:{'expected':expected,'observed':metrics[key]['value']}
                for key,expected in fields.items() if key not in missing and metrics[key]['value'] != expected}
    return {'status':'pass' if not missing and not mismatched else 'unverified' if missing and not mismatched else 'fail',
            'missing':missing,'mismatched':mismatched,
            'scope':'kernel launch grid/block/register count; not tensor shape or kernel numerical correctness'}
