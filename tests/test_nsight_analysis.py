import tempfile
from pathlib import Path
import unittest
import sqlite3
from scripts.nsight_analysis import interval_union_ns, read_systems, read_compute_csv, compute_targets, verify_compute_launch


class NsightAnalysisTests(unittest.TestCase):
    def test_compute_launch_requires_actual_metadata(self):
        selection={'grid':[2,1,1],'block':[64,1,1],'registers_per_thread':32}
        self.assertEqual(verify_compute_launch({},selection)['status'],'unverified')
        metrics={f'launch__{kind}_dim_{axis}':{'value':selection[kind][i]}
                 for kind in ['grid','block'] for i,axis in enumerate('xyz')}
        metrics['launch__registers_per_thread']={'value':32}
        self.assertEqual(verify_compute_launch(metrics,selection)['status'],'pass')
        metrics['launch__grid_dim_x']['value']=1
        self.assertEqual(verify_compute_launch(metrics,selection)['status'],'fail')
    def test_launch_variants_and_nested_pipeline_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('''
                    create table StringIds(id integer, value text);
                    insert into StringIds values(1, 'same_gemm');
                    create table NVTX_EVENTS(start integer, end integer, text text, textId integer);
                    insert into NVTX_EVENTS values(0,1000,'esm3_capture',null);
                    insert into NVTX_EVENTS values(10,900,'esm3_phase_forward_0',null);
                    create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,demangledName integer,
                        streamId integer,gridX integer,gridY integer,gridZ integer,
                        blockX integer,blockY integer,blockZ integer,registersPerThread integer);
                    insert into CUPTI_ACTIVITY_KIND_KERNEL values(20,120,1,1,1,1,1,32,1,1,16);
                    insert into CUPTI_ACTIVITY_KIND_KERNEL values(200,700,1,1,4,1,1,64,1,1,32);
                ''')
            data = read_systems(path)
            self.assertEqual(len(data['kernels']), 1)
            self.assertEqual(len(data['kernel_launch_variants']), 2)
            self.assertEqual(data['kernel_launch_variants'][0]['grid'], [4,1,1])
            self.assertAlmostEqual(data['kernel_launch_variants'][0]['kernel_sum_percent'], 500/600*100)
            self.assertEqual(data['iterations'][0]['name'], 'esm3_phase_forward_0')
            self.assertAlmostEqual(data['gpu_activity_fraction_percent'], 60.)
            target = compute_targets(data, 1)[0]
            self.assertEqual(target['filtered_launch_skip'],1)
            self.assertEqual(target['grid'],[4,1,1])
            # Model-specific ranges must be selected explicitly; a CUDA trace
            # from another workload must not silently inherit ESM3's scope.
            with sqlite3.connect(path) as db:
                db.execute("update NVTX_EVENTS set text='openfold_capture' where text='esm3_capture'")
                db.execute("update NVTX_EVENTS set text='openfold_phase_task_0' where text='esm3_phase_forward_0'")
            with self.assertRaisesRegex(ValueError, 'esm3_capture'):
                read_systems(path)
            data = read_systems(path, capture_name='openfold_capture')
            self.assertEqual(data['capture_name'], 'openfold_capture')
            self.assertEqual(compute_targets(data, 1)[0]['filtered_launch_skip'], 1)
    def test_overlap_is_not_double_counted(self):
        self.assertEqual(interval_union_ns([(1,7),(3,9),(9,11)],0,12),10)

    def test_clip_to_capture_and_exclude_invalid(self):
        self.assertEqual(interval_union_ns([(-5,3),(8,20),(4,4),(7,2)],0,10),5)

    def test_empty_activity_is_empty_union(self):
        self.assertEqual(interval_union_ns([],0,10),0)

    def test_compute_missing_counters_remain_null(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'metrics.csv'
            p.write_text('preamble\n"ID","Metric Name","Metric Unit","Metric Value"\n"0","launch__registers_per_thread","register/thread","32"\n')
            r=read_compute_csv(p)
            self.assertIsNone(r['selected_metrics']['occupancy_percent'])
            self.assertEqual(r['selected_metrics']['registers_per_thread']['value'],32)

    def test_compute_thousands_separator_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'metrics.csv'
            p.write_text('"Metric Name","Metric Unit","Metric Value"\n"dram__bytes_read.sum","byte","1,024"\n')
            r=read_compute_csv(p)
            self.assertEqual(r['selected_metrics']['dram_read_bytes']['value'],1024)
            self.assertEqual(r['selected_metrics']['dram_read_bytes']['unit'],'byte')

    def test_denied_compute_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'denied.csv';p.write_text('==ERROR== ERR_NVGPUCTRPERM')
            with self.assertRaisesRegex(ValueError,'no metric header'):read_compute_csv(p)

    def test_multiple_launches_are_not_silently_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'multi.csv'
            p.write_text('"ID","Metric Name","Metric Unit","Metric Value"\n"0","x","%","10"\n"1","x","%","20"\n')
            with self.assertRaisesRegex(ValueError,'Multiple Compute launches'):read_compute_csv(p)

    def test_empty_sqlite_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'empty.sqlite';p.touch()
            with self.assertRaisesRegex(ValueError,'Missing required'):read_systems(p)
