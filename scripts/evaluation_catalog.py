"""Build an evidence-backed experiment index without running inference."""
import ast
from collections import Counter
from html import escape
import json
import os
from pathlib import Path

from common.reporting import sha256, write_json

REPO = Path(__file__).resolve().parents[1]
GROUPS = {
    'core': 'ESM3 / 核心任务与长度边界',
    'standard': 'ESM3 / 标准序列性能与真实 batch',
    'diagnostic': '数值一致性 / CPU 与 padding 对照',
    'fitness': 'ESM3 / 实验功能评分 DMS',
    'openfold': 'OpenFold / 完整参考预测',
    'operator': '算子 / shape 驱动简化诊断',
    'systems': 'Nsight Systems / 原生时间线',
}
TASK_IO = {
    'forward': ('已知序列；无实验坐标', '全位置序列 logits + embeddings；不是生成序列'),
    'sequence': ('部分序列；指定位置设为 _；无坐标条件', '补全后的序列'),
    'structure': ('完整序列；不给实验坐标', '结构 token → 原子坐标 / PDB'),
    'inverse': ('部分序列 + 实验 PDB A 链坐标', '带结构条件的序列补全；不是全序列逆折叠'),
}


def generation_settings(report, root, sources):
    """Recover omitted defaults only from a hash-matched inference source."""
    if report['task'] == 'forward':
        return None
    expected = report['environment']['source_sha256']['models/esm3_adapter.py']
    candidates = [root / 'esm3/nsight/source/models/esm3_adapter.py', REPO / 'models/esm3_adapter.py']
    source = next((p for p in candidates if p.is_file() and sha256(p) == expected), None)
    if source is None:
        return {'status': 'unresolved', 'note': '历史源码无法核对，不用当前默认值猜测生成设置'}
    sources[str(source)] = expected
    tree = ast.parse(source.read_text())
    prepare = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    call = next(n for n in ast.walk(prepare) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'GenerationConfig')
    static = {k.arg: ast.literal_eval(k.value) for k in call.keywords if isinstance(k.value, ast.Constant)}
    branch = next(n for n in ast.walk(prepare) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                  and n.test.left.id == 'task' and any(isinstance(c, ast.Constant) and c.value == 'structure'
                                                       for c in n.test.comparators))
    defaults = next(n.value.elts[1].values[-1].value for n in branch.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Tuple))
    cfg = report['config']
    masks = [v['position'] for v in report['quality'].get('masked_positions', [])]
    steps = cfg.get('steps') or (defaults if report['task'] == 'structure' else len(masks))
    return dict(static, track='structure' if report['task'] == 'structure' else 'sequence',
                num_steps=steps, masked_positions=masks, coordinate_condition=report['task'] == 'inverse',
                source=str(source), source_sha256=expected,
                origin='effective settings recovered from hash-matched historical source + recorded config')


def build_catalog(root, esm, panel, fitness, batch, of, ns, ons, op, micro, read, sources):
    root = Path(root)
    records = []

    def add(id, group, model, task, batch_size, lengths, precision, device, settings, source,
            input_description, output_description, status, warmup=None, runs=None,
            timing_scope='', evidence=None, batch_unit='蛋白请求', token_width=None):
        records.append(dict(id=id, group=group, model=model, task=task, batch_size=batch_size,
                            batch_unit=batch_unit, residue_lengths=lengths, token_width=token_width,
                            precision=precision, device=device, warmup=warmup, runs=runs,
                            timing_scope=timing_scope, settings=settings, source=str(Path(source)),
                            input=input_description, output=output_description, status=status,
                            evidence=[str(Path(p)) for p in (evidence or [source])]))

    for folder, summary, group in [('esm3', esm, 'core'), ('esm3_standard', panel, 'standard')]:
        for name, value in summary['cases'].items():
            report = value['report']; cfg = report['config']; lengths = report['residue_lengths']
            source = root / folder / name / 'report.json'
            sources[str(source)] = sha256(source)
            gen = generation_settings(report, root, sources)
            settings = dict(seed=cfg['seed'], attention_requested=cfg['attention'], cpu_threads=cfg['cpu_threads'],
                            sample_ids=report['sample_ids'], input_file=cfg['inputs'],
                            generation=gen, precision_policy=report['precision'],
                            timing_exclusions=report['timing_exclusions'], hot_delivery_runs=cfg['end_to_end_runs'])
            if gen and report['task'] in ('structure', 'inverse'):
                settings['pdb'] = cfg['pdb']
                settings['pdb_use'] = '输入结构条件与质量参照' if report['task'] == 'inverse' else '仅输出质量参照；不进入生成输入'
            add(f'{folder}/{name}', group, 'ESM3', report['task'], cfg['batch_size'], lengths,
                cfg['precision'], cfg['device'], settings, source, *TASK_IO[report['task']],
                report['status'], cfg['warmup'], report['latency']['count'], report['timing_scope'],
                token_width=max(lengths) + 2 if report['task'] == 'forward' else None)

    add('diagnostic/esm3_cpu_cuda', 'diagnostic', 'ESM3', 'forward', 1,
        esm['cases']['ubiquitin_fp32']['report']['residue_lengths'], 'fp32', 'cpu ↔ cuda:0',
        dict(atol=.001, rtol=.001, compared='所有有效 logits / embeddings，排除 BOS/EOS/padding',
             cpu_threads=esm['cases']['ubiquitin_cpu_fp32']['report']['config']['cpu_threads']),
        root/'esm3/audit_summary.json', '同一泛素序列与 tokens', 'CPU/CUDA 完整有效张量误差',
        esm['comparisons']['cpu_cuda_fp32']['status'], timing_scope='数值比较；不使用 profiler 时间',
        evidence=[root/'esm3/audit_summary.json', root/'esm3/ubiquitin_cpu_fp32/report.json', root/'esm3/ubiquitin_fp32/report.json'])
    add('diagnostic/fp32_padding_batch', 'diagnostic', 'ESM3', 'forward', [1, 4], batch['lengths'],
        'fp32', batch['parameters']['device'], dict(thresholds=batch['thresholds'], seed=7,
        sample_ids=batch['parameters']['sample_ids'], padding_width=max(batch['lengths'])+2,
        comparisons=['原长 B1 ↔ padding B1', '原长 B1 ↔ padding B4', '改变 padding token 内容后有效输出']),
        root/'esm3_batch_fp32/batch_diagnostic.json', '四个固定标准 target；原长或补齐到相同宽度',
        '全有效 logits / embeddings 与 padding 隔离检查', batch['status'],
        timing_scope='数值诊断；不是正式性能样本')
    bfcheck=next(iter(panel['batch_consistency'].values()))
    add('diagnostic/bf16_B1_B4', 'diagnostic', 'ESM3', 'forward', [1, 4], batch['lengths'],
        'bf16', panel['cases']['mixed_B4']['report']['config']['device'],
        dict(atol=bfcheck['atol'],rtol=bfcheck['rtol'],
             sample_ids=list(panel['batch_consistency']), padding_width=max(batch['lengths'])+2,
             failed_checks=sum(v['status']!='pass' for v in panel['batch_consistency'].values()),
             compared='相同精度下原长B1与变长padding B4的完整有效logits/embeddings'),
        root/'esm3_standard/panel_summary.json','四个固定标准target；B1与padding B4',
        '逐target数值误差；执行PASS与比较失败分别保存',
        'pass' if all(v['status']=='pass' for v in panel['batch_consistency'].values()) else 'fail',
        timing_scope='从正式运行保存的输出做完整数值比较；不另造性能样本')

    for name, report in fitness['assays'].items():
        cfg=fitness['parameters']; n=report['masked_contexts']; cap=cfg['batch_size']
        effective=[min(cap,n-i) for i in range(0,n,cap)]
        add(f'esm3_fitness/{name}', 'fitness', 'ESM3', 'masked_forward', cap, [report['target_residues']],
            cfg['precision'], cfg['device'], dict(seed=7, context_batch_sizes=effective,
            masked_positions=report['positions'], masked_contexts=n, variant_rows=report['rows'],
            scoring=fitness['scoring'], mutation_mode=report['metadata']['includes_multiple_mutants'],
            probe=report['batch_check'], cpu_threads=cfg['cpu_threads']),
            root/'esm3_fitness/dms_summary.json', '完整 WT context；每个 context 只 mask 一个实验位置',
            '突变 fitness 评分 CSV；Spearman / AUC', report['status'], 1, report['latency_seconds']['count'],
            report['timing_scope'], evidence=[root/'esm3_fitness/dms_summary.json', root/'esm3_fitness'/name/'predictions.csv'],
            batch_unit='masked 上下文；不是变体数', token_width=report['target_residues']+2)

    of_settings=dict(seed=of['parameters']['seed'], cpu_threads=of['parameters']['cpu_threads'],
        model=of['model'], chunk_size=of['config']['globals']['chunk_size'],
        evoformer_blocks=of['config']['model']['evoformer_stack']['no_blocks'],
        extra_msa_blocks=of['config']['model']['extra_msa']['extra_msa_stack']['no_blocks'],
        recycles=of['config']['data']['common']['max_recycling_iters'], model_passes=of['output']['num_model_passes'],
        msa_shape=of['features']['shapes']['msa_feat'], extra_msa_shape=of['features']['shapes']['extra_msa'],
        templates=of['config']['data']['common']['use_templates'], relaxation=False,
        attention=of['attention_implementation'], fasta=of['parameters']['fasta'], alignments=of['parameters']['alignments'])
    add('openfold/6KWC', 'openfold', 'OpenFold', 'structure', 1, [of['output']['residues']], 'fp32',
        of['parameters']['device'], of_settings, root/'openfold/report.json',
        '6KWC 单体序列 + 预计算真实 MSA；无模板', '完整原子坐标 / pLDDT / 未松弛 PDB', of['status'],
        of['parameters']['warmup'], of['latency_seconds']['count'], of['timing_scope'],
        evidence=[root/'openfold/report.json',root/'openfold/structure_quality.json',root/'openfold/prediction_unrelaxed.pdb'])
    comp_path=root/'openfold_cpu_comparison.json'
    if comp_path.exists():
        comp=read(comp_path)
        add('diagnostic/openfold_cpu_cuda', 'diagnostic', 'OpenFold', 'structure', 1,
            [of['output']['residues']], 'fp32', 'cpu ↔ cuda:0', dict(reference=str(comp.get('reference')),
            thresholds=comp.get('thresholds'),
            branch_difference='CPU 官方 torch Attention / IPA；CUDA upstream 扩展'), comp_path,
            '同一冻结 MSA features / 权重 / 配置', '全有效原子坐标与 pLDDT 误差', comp['status'],
            timing_scope='复用真实 CPU 完整参考做数值比较；不是新 CPU 性能实验')

    micro_source=next((p for p in [REPO/'scripts/portable_operator_bench.py'] if p.is_file()
                       and sha256(p)==micro['environment']['source_sha256']['scripts/portable_operator_bench.py']), None)
    # Historical microbench files omit warmup; do not guess after a source change.
    warmup=None
    if micro_source:
        tree=ast.parse(micro_source.read_text());ranges=[n.iter for n in ast.walk(tree) if isinstance(n,ast.For)]
        constants=[ast.literal_eval(n.args[0]) for n in ranges if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Name) and n.func.id=='range' and isinstance(n.args[0],ast.Constant)]
        warmup=constants[0] if len(constants)==1 else None
        sources[str(micro_source)]=sha256(micro_source)
    for v in micro['results']:
        add('operator/'+v['operator'], 'operator', '算子', v['operator'], None, None,
            micro['precision'], 'cuda:'+str(micro['device']['index']),
            dict(input_shapes=v['input_shapes'], input_strides=v['input_strides'], input_dtypes=v['input_dtypes'],
            source_shapes=micro['source_shapes'], input_seed=7, simplified=True,
            warmup_source=str(micro_source) if micro_source else None),
            root/'operator_diagnostics/microbench.json', 'CPU 固定种子合成输入；shape 来自模型 hooks',
            '算子 wall/Event 延迟、显存及 FP32 描述性误差', 'measured', warmup, v['latency']['count'],
            '同步单算子调用；不是完整蛋白推理', batch_unit='算子 tensor shape；不按蛋白定义 B/L')

    for folder, summary in [('esm3/nsight', ns), ('esm3_standard/nsight', None)]:
        if summary is None:
            p=root/folder/'nsight_summary.json'
            if not p.exists():continue
            summary=read(p)
        for name, capture in summary['systems'].items():
            wpath=root/folder/capture['workload']; w=read(wpath); cfg=w['config']
            add(folder+'/'+name, 'systems', 'ESM3', cfg['task'], cfg['batch_size'], w['residue_lengths'],
                cfg['precision'], cfg['device'], dict(seed=cfg['seed'], attention_requested=cfg['attention'],
                capture_scope=w['scope'], mask_positions=cfg['mask_positions'] if cfg['task'] in ('sequence','inverse') else [],
                sample_ids=cfg['sample_ids'] or [cfg['sample_id']],
                generation=generation_settings(dict(task=cfg['task'],config=cfg,environment=w['environment'],quality=w['quality']),root,sources),
                cpu_threads=cfg['cpu_threads'], trace=['CUDA','NVTX','OSRT'], cpu_sampling=False,
                cpu_context_switch=False, reference=w['reference']), wpath, *TASK_IO[cfg['task']], capture['status'],
                cfg['warmup'] if w['scope']=='warm' else 0, cfg['calls'], w['scope_definition'],
                evidence=[wpath, root/folder/capture['native_report']],
                token_width=max(w['residue_lengths'])+2 if cfg['task']=='forward' else None)
    for folder, capture in [('openfold_nsight',ons), ('openfold_pipeline',op)]:
        wpath=root/folder/'workload/workload.json'; w=read(wpath)
        command=read(root/folder/capture['capture']['command'])['argv']; scope=w['capture']
        warm=int(command[command.index('--warmup')+1]) if scope=='warm' else 0
        add(folder+'/6KWC', 'systems', 'OpenFold', 'structure', 1, [of['output']['residues']], 'fp32',
            of['parameters']['device'], dict(of_settings,capture_scope=scope,
            trace=['CUDA','NVTX','OSRT'],cpu_sampling=False,cpu_context_switch=False), wpath,
            '6KWC + 预计算 MSA；warm 用冻结 features，pipeline 实际重新 prepare',
            '完整预测时间线；pipeline 额外包含加载/准备/CPU-PDB交付', capture['status'], warm,
            sum(p['name'].startswith('task_') for p in w['phases']), w['scope_note'],
            evidence=[wpath, Path(capture['native_report'])])

    counts=dict(Counter(v['group'] for v in records))
    catalog=dict(schema_version=1, scope='RTX4090 reference evaluation; no new model runs',
        counts=counts, experiment_count=len(records), records=records,
        conventions=dict(B='本次模型调用同时输入的蛋白数；DMS 改为 masked context 数，逐行说明',
        L='残基数，不包含 ESM3 BOS/EOS；forward tensor width 为 max(L)+2',
        statuses='模型执行 PASS、科学质量、数值一致性、profiler 权限、交付核验分别判断',
        source='原报告优先；生成默认值只从 SHA 匹配的源码恢复；未知值保留 null'))
    write_json(root/'experiment_index.json',catalog)
    return catalog


def render_catalog(catalog, root, link):
    blocks=[]
    for group,label in GROUPS.items():
        rows=[]
        for r in catalog['records']:
            if r['group']!=group:continue
            lengths=' / '.join(map(str,r['residue_lengths'])) if r['residue_lengths'] else '见 tensor shape'
            b=str(r['batch_size']) if r['batch_size'] is not None else '—'
            gen=r['settings'].get('generation')
            description=r['input']+' → '+r['output']
            if gen and 'num_steps' in gen:
                description+=f'；track={gen["track"]}，steps={gen["num_steps"]}，T={gen.get("temperature")}，mask={gen["masked_positions"] or "—"}'
            if r['group']=='fitness':
                description+=f'；实际各 batch={r["settings"]["context_batch_sizes"]}；{r["settings"]["variant_rows"]:,}变体'
            values=escape(json.dumps(r['settings'],ensure_ascii=False,indent=2))
            timing=f'预热 {r["warmup"] if r["warmup"] is not None else "不适用 / 未记录"}；正式 / 捕获 {r["runs"] if r["runs"] is not None else "不适用"}'
            evidence=' · '.join(link(Path(p),Path(p).name) for p in r['evidence'])
            rows.append(f'<tr data-experiment="{escape(r["id"],quote=True)}"><td><code>{escape(r["id"])}</code><br><span class="muted">{escape(r["status"])}</span></td><td>{escape(r["task"])}<br>{escape(r["precision"])} / {escape(r["device"])}</td><td>B={escape(b)}<br>L={escape(lengths)}<br><small>{escape(r["batch_unit"])}</small></td><td>{escape(description)}<br><small>{escape(timing)}</small><details><summary>全部 settings / 计时范围</summary><p>{escape(r["timing_scope"])}</p><pre>{values}</pre></details>{evidence}</td></tr>')
        blocks.append(f'<details class="catalog-group"><summary>{escape(label)} · {len(rows)} 项</summary><div class="scroll"><table class="settings-table"><tr><th>实验 ID / 执行状态</th><th>任务 / 精度 / 设备</th><th>Batch / Length</th><th>输入 → 输出 / 次数 / 证据</th></tr>{"".join(rows)}</table></div></details>')
    return '<section id="settings"><h2>实验设置索引 / 每一项对应实际产物</h2><p>展开类别查看每个实验；“全部 settings”显示 seed、生成参数、精度策略、MSA、输入路径与计时范围。编号采用目录/实验名，可直接定位文件。这里的数量包括数值诊断与 profiling，不能当作同口径性能样本数相加。</p>'+''.join(blocks)+'</section>'


def render_overview(catalog, esm, panel, fitness, of, systems_count, link, root):
    core=[v for v in catalog['records'] if v['group']=='core']
    standard=[v for v in catalog['records'] if v['group']=='standard']
    lengths=sorted(v['residue_lengths'][0] for v in standard if v['batch_size']==1)
    ofr=next(v for v in catalog['records'] if v['group']=='openfold')
    frows=[v for v in catalog['records'] if v['group']=='fitness']
    env=esm['cases']['L256']['report']['environment']
    oef=of['environment']; mixed=panel['cases']['mixed_B4']['report']['residue_lengths']
    bf_fail=sum(v['status']!='pass' for v in panel['batch_consistency'].values())
    def cell(title, body):
        return '<article class="map-cell"><h4>'+title+'</h4>'+body+'</article>'
    core_bench=f'<b>{len(core)} 个正式任务配置</b><p>forward：合成 L=256/1024/2048，B1；真实混合 B2，L=21/76。生成：泛素 L76、crambin L46，B1，sequence / structure / inverse。</p><p>sequence / inverse遮住4、5、13位，生成3步；structure生成8步，温度0.7。BF16主干；CPU↔CUDA forward 对照另用FP32。预热3 / 正式20；CPU正式3。</p>'
    standard_body=f'<b>{len(standard)} 个标准性能配置</b><p>10 个 ProteinGym target，forward / B1 / BF16；L={escape(str(lengths))}。</p><p>混合 B4：L={escape(str(mixed))}；FP32 padding / batch 另做数值诊断。正式配置预热3 / 测量20。</p>'
    variants=sum(v['rows'] for v in fitness['assays'].values())
    fitness_body='<b>3 个完整 DMS assay</b><p>masked-forward / FP32；B≤8个上下文（SPG1实际B4），不是8个变体。</p><p>L='+escape('/'.join(str(v['residue_lengths'][0]) for v in frows))+f'；每assay预热1次完整sweep / 正式3次；评分全部{variants:,}变体。</p>'
    ofs=ofr['settings']
    of_body=f'<b>1 个完整参考配置</b><p>6KWC / B1 / L={ofr["residue_lengths"][0]} / structure / FP32。预计算MSA、无模板、未松弛。</p><p>{ofs["evoformer_blocks"]} Evoformer + {ofs["extra_msa_blocks"]} ExtraMSA blocks；recycling={ofs["recycles"]} → {ofs["model_passes"]}遍主干；chunk{ofs["chunk_size"]}；预热{ofr["warmup"]} / 正式{ofr["runs"]}。</p><p>MSA tensor {escape(str(ofs["msa_shape"]))}；ExtraMSA {escape(str(ofs["extra_msa_shape"]))}，末维是循环遍数，不是batch。</p>'
    return f'''<section id="overview"><h2>00 / 全局概览：先看做了哪些实验</h2>
<p>下面按实验依赖分四层阅读。第一层固定条件，第二层分模型与输入，第三层列具体调用，第四层分别判断性能、正确性、质量与热点。纵向连线表示组织关系，不表示这些实验串成一次模型推理。</p>
<div class="experiment-map" role="group" aria-label="4090评测四层总览">
<div class="map-level"><span class="level-mark">01 / 固定条件</span><div class="map-shared"><b>单张 RTX 4090 · 两个独立环境 · 固定源码 / 权重 / 输入</b><p>ESM3：Python {escape(env['python'].split()[0])} / Torch {escape(env['versions']['torch'])} / esm {escape(env['versions']['esm'])} / seed7。OpenFold：Python {escape(oef['python'].split()[0])} / Torch {escape(oef['versions']['torch'])} / seed42。TF32关闭；辅助模块精度单独记录。</p></div></div>
<div class="map-level"><span class="level-mark">02 / 模型与输入</span><div class="map-models"><div class="model-title">ESM3 · 序列 / 实验结构 / ProteinGym</div><div class="model-title">OpenFold · 6KWC 序列 + 真实 MSA</div></div></div>
<div class="map-level"><span class="level-mark">03 / 任务 settings</span><div class="map-experiments"><div class="esm-branches">{cell('核心路径与边界',core_bench)}{cell('标准长度与 batch',standard_body)}{cell('变体功能评分',fitness_body)}</div>{cell('完整单体结构预测',of_body)}</div></div>
<div class="map-level"><span class="level-mark">04 / 指标与证据</span><div class="map-evidence">{cell('正式性能','<p>加载 / 首调用 / 稳态median、P90、CV / 吞吐 / 显存；与profiling分开。B与L不同的任务不混排行。</p>')}{cell('数值与任务质量','<p>CPU完整输出、B1↔B4、padding隔离；掩码恢复、实验结构RMSD/lDDT、DMS相关与AUC。</p>')}{cell('热点与阶段','<p>ATen / 蛋白模块shape；8个简化算子；'+str(systems_count)+'份原生Systems，warm与pipeline分别看；Compute权限不足。</p>')}</div></div></div>
<div class="reading pending"><b>覆盖结论：</b>当前冻结的参考路径已实测；BF16变长batch有{bf_fail}/4项超阈值；Compute没有硬件计数器。OpenFold长度/模型batch扫描、模板/搜索/relaxation、更多独立结构、A100/S5000及多卡均未实测。此图说明已做范围，不表示所有分支验收通过。</div>
<div class="setting-glossary"><p><b>BATCH（B）</b>：一次模型调用并排处理的输入数。B4 forward是真批次；逐个生成四次仍是B1。DMS的输入单位改为masked context，明确标注。</p><p><b>LENGTH（L）</b>：每条蛋白的残基数。ESM3 forward另加BOS/EOS，因此L256实际token宽258；变长B4统一padding到max(L)+2=854。padding不算有效残基。</p><p><b>任务与输出</b>：forward输出logits/embeddings；sequence补序列；structure产坐标；inverse以坐标为条件补序列。DMS复用masked-forward评分，OpenFold直接做完整结构预测。</p><p><b>计时三类</b>：benchmark是正式稳态调用；hot delivery增加准备与CPU/PDB交付；Systems pipeline在新进程抓加载到交付，带插桩开销，不代表磁盘冷缓存，也不替代benchmark。</p></div>
<p><a href="#settings">逐实验 settings</a> · <a href="#layout">目录层级与命名</a> · {link(root/'experiment_index.json','机器可读实验索引')} · {link(root/'RESULTS.md','结果文件导航')}</p></section>'''


def write_results_guide(catalog, root):
    layout=[('esm3/', '核心任务、合成长度边界、B2、CPU对照；每个case独立目录'),
            ('esm3_standard/', '十个标准 target 的B1与混合B4；panel_summary保留BF16比较失败'),
            ('esm3_batch_fp32/', 'B1 / padding B1 / B4 / padding内容隔离的FP32诊断'),
            ('esm3_fitness/', '三个独立assay；predictions.csv是逐变体评分，score_tables.pt是复用的上下文输出'),
            ('openfold/', '正式预测、features、完整输出、PDB、结构质量、ATen与蛋白模块'),
            ('esm3/nsight/、esm3_standard/nsight/', 'ESM3原生warm/pipeline捕获；与正式报告相互引用'),
            ('openfold_nsight/、openfold_pipeline/', 'OpenFold原生warm与pipeline；Compute权限与计划保存在summary'),
            ('operator_diagnostics/', '简化算子诊断，不等同真实模型激活重放'),
            ('figures/', '由实测数据生成的可视化；来源hash在report_provenance.json')]
    lines=['# 4090 评测结果导航','',
        '先打开 [report.html](report.html)；逐实验设置见 [experiment_index.json](experiment_index.json)，正式性能指标见 [metrics.csv](metrics.csv)。',
        '', '## 层级与文件职责','', '| 路径 | 内容 |','|---|---|']
    lines += [f'| `{p}` | {text} |' for p,text in layout]
    lines += ['', '## 命名规则与历史短名','',
        '`L256` = ESM3 forward / 合成random_L256 / B1 / BF16；`sequence`、`structure`、`inverse` = 泛素76残基 / B1；`crambin_*` = 1CRN46残基。',
        '`mixed_B2` = 胰岛素A链21 + 泛素76；`mixed_B4` = 标准target47/237/413/852。它们均是真forward批次，生成不使用B2/B4。',
        '`*_warm` = 预热后操作范围；`*_pipeline` = 新进程加载/准备/首次任务/交付范围。相同名字位于不同目录时，必须携带完整相对路径引用。',
        '原始目录与文件路径已被原生报告、记录的argv及hash引用，因此保留这些路径；索引补足其含义，不篡改历史报告。',
        '', '## 全部实验','', '| ID | 任务 | B | L/残基 | 精度 | 原始证据 |','|---|---|---|---|---|---|']
    for r in catalog['records']:
        path=os.path.relpath(r['source'],root)
        b=r['batch_size'] if r['batch_size'] is not None else '见tensor shape'
        lengths=r['residue_lengths'] if r['residue_lengths'] is not None else '不按蛋白L定义'
        lines.append(f'| `{r["id"]}` | {r["task"]} | {b} | {lengths} | {r["precision"]} | [{Path(path).name}]({path}) |')
    lines += ['', '## 如何判断结果','',
        '1. 正式性能只比较相同任务/输入/B/L/精度/计时范围的latency；Systems插桩时长另看。',
        '2. 模型执行PASS不等于数值一致性PASS或功能质量达标；BF16批次失败与Compute缺失保留。',
        '3. OpenFold此次限无模板、预计算MSA、未松弛单体；A100/S5000和未测分支不能据此宣称通过。',
        '4. `verification.json`检查交付完整性；`execution.json`记录实际阶段退出状态；`cleanup_manifest.json`记录已授权清理。','']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
