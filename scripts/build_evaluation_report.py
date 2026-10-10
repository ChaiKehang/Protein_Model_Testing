"""Explain measured ESM3/OpenFold reference evidence, including failed diagnostics."""
import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from common.reporting import sha256,write_json
from scripts.evaluation_catalog import build_catalog, render_catalog, render_overview, write_results_guide


def main(a):
 root=Path(a.output).resolve();sources={}
 def read(path):
  p=Path(path);sources[str(p)]=sha256(p);return json.loads(p.read_text())
 def link(path,label):return '<a href="'+escape(os.path.relpath(path,root))+'">'+escape(label)+'</a>'
 def image(name,caption):return '<figure><img src="figures/'+name+'.svg" alt="'+escape(caption)+'"><figcaption>'+caption+'</figcaption></figure>'
 esm=read(root/'esm3/audit_summary.json');panel=read(root/'esm3_standard/panel_summary.json');fitness=read(root/'esm3_fitness/dms_summary.json')
 batch=read(root/'esm3_batch_fp32/batch_diagnostic.json');of=read(root/'openfold/report.json');quality=read(root/'openfold/structure_quality.json')
 ns=read(root/'esm3/nsight/nsight_summary.json');ons=read(root/'openfold_nsight/systems_summary.json');op=read(root/'openfold_pipeline/systems_summary.json')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 from scipy.stats import rankdata
 from matplotlib.colors import LogNorm
 from scripts.figure_style import report_style
 figdir=root/'figures';figdir.mkdir(exist_ok=True)
 cases=esm['cases'];standard=[(k,v['report']) for k,v in panel['cases'].items() if k!='mixed_B4']
 standard.sort(key=lambda kv:kv[1]['residue_lengths'][0]);tasks=['sequence','structure','inverse','crambin_sequence','crambin_structure','crambin_inverse']
 systems=[(k,v['analysis'],root/'esm3/nsight'/v['native_report']) for k,v in ns['systems'].items() if v['status']=='pass']
 extra_path=root/'esm3_standard/nsight/nsight_summary.json'
 if extra_path.exists():
  extra=read(extra_path)
  systems += [(k,v['analysis'],extra_path.parent/v['native_report']) for k,v in extra['systems'].items() if v['status']=='pass']
 systems += [('OpenFold_warm',ons['analysis'],Path(ons['native_report'])),('OpenFold_pipeline',op['analysis'],Path(op['native_report']))]
 @report_style
 def plots():
  fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
  x=[r['residue_lengths'][0] for k,r in standard]
  for ax,ys,label in [(axes[0],[r['latency']['median']*1000 for k,r in standard],'同步稳态 median / ms'),(axes[1],[r['memory']['allocated_peak_gib'] for k,r in standard],'allocated peak / GiB')]:
   ax.scatter(x,ys,s=55,color='#087F76',marker='o',zorder=3);ax.set(xlabel='标准 target 长度 / 残基',ylabel=label,xlim=(0,2000),ylim=(0,max(ys)*1.2));ax.grid(axis='y')
  axes[0].set_title('10 个实际输入点 · 不拟合',loc='left');axes[1].set_title('包含常驻模型与输入/输出',loc='left');fig.savefig(figdir/'lengths.svg');fig.savefig(figdir/'lengths.png',dpi=170);plt.close(fig)
  fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');labels=['forward','sequence','structure','inverse']
  for j,(field,col,marker,label) in enumerate([('first_operation_seconds','#A35E20','D','首调用'),('latency','#087F76','o','稳态'),('end_to_end','#365D9A','s','热交付')]):
   vals=[]
   for name in ['L256','sequence','structure','inverse']:
    v=cases[name]['report'][field];vals.append(v['median'] if isinstance(v,dict) else v)
   ax.scatter(vals,np.arange(4)+(j-1)*.15,c=col,marker=marker,s=45,label=label,zorder=3)
  ax.set(xscale='log',xlabel='同步时间 / s · 对数轴',yticks=range(4),yticklabels=labels);ax.invert_yaxis();ax.grid(axis='x');ax.legend();fig.savefig(figdir/'task_timing.svg');plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained');ids=list(panel['batch_consistency'])
  for ax,key in zip(axes,['logits','embeddings']):
   bf=[panel['batch_consistency'][id]['metrics'][key]['relative_l2'] for id in ids];fp=[batch['checks'][id]['unpadded_B1_vs_B4'][key]['metrics'][key]['relative_l2'] for id in ids]
   pos=np.arange(len(ids));ax.bar(pos-.17,bf,width=.3,label='BF16',color='#75869D');ax.bar(pos+.17,fp,width=.3,label='FP32',color='#087F76');ax.set(yscale='log',xticks=pos,xticklabels=[id[3:].replace('_','\n',1) for id in ids],ylabel='relative L2 · 对数轴');ax.set_title(key,loc='left');ax.legend();ax.grid(axis='y')
  fig.savefig(figdir/'batch_errors.svg');plt.close(fig)
  hs=[];rr=[]
  for name,r in fitness['assays'].items():
   data=list(csv.DictReader((root/'esm3_fitness'/name/'predictions.csv').open()));y=np.array([float(x['DMS_score']) for x in data]);p=np.array([float(x['esm3_masked_marginal_score']) for x in data]);yr=(rankdata(y)-.5)/len(y);pr=(rankdata(p)-.5)/len(p);h,xe,ye=np.histogram2d(yr,pr,bins=40,range=[[0,1],[0,1]]);hs.append((h/len(y),xe,ye));rr.append((name,r))
  fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained');norm=LogNorm(vmin=min(float(h[h>0].min()) for h,_,_ in hs),vmax=max(float(h.max()) for h,_,_ in hs))
  for ax,(h,xe,ye),(name,r) in zip(axes,hs,rr):
   mesh=ax.pcolormesh(xe,ye,np.ma.masked_where(h.T==0,h.T),norm=norm,cmap='viridis',rasterized=True);ax.plot([0,1],[0,1],'--',color='white',lw=1,alpha=.6);ax.set(xlabel='实验 fitness 排名分位',ylabel='模型评分排名分位',aspect='equal');ax.set_title(name.split('_')[0]+f' · n={r["rows"]:,}\nSpearman={r["quality"]["spearman"]:.3f}',loc='left')
  fig.colorbar(mesh,ax=axes,label='每格占该实验的样本比例 · 对数色标',shrink=.7);fig.savefig(figdir/'fitness.svg');fig.savefig(figdir/'fitness.png',dpi=170);plt.close(fig)
  selected=[v for v in systems if 'pipeline' not in v[0]]
  fig,ax=plt.subplots(figsize=(11,5),layout='constrained');pos=np.arange(len(selected));left=np.zeros(len(selected))
  for i,col in enumerate(['#087F76','#365D9A','#A35E20','#D7E0E7']):
   vals=[sum(k['kernel_sum_percent'] for k in an['kernels'][3:]) if i==3 else an['kernels'][i]['kernel_sum_percent'] for name,an,p in selected];ax.barh(pos,vals,left=left,height=.6,color=col,label='其余' if i==3 else f'排名 K{i+1}')
   left+=np.array(vals)
  ax.set(xlim=(0,100),yticks=pos,yticklabels=[v[0] for v in selected],xlabel='kernel 累计时间占比 / %');ax.invert_yaxis();ax.legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.01));fig.savefig(figdir/'hotspots.svg');plt.close(fig)
  phase_groups=[('ESM3 inverse',ns['systems']['inverse_pipeline']['analysis']),('OpenFold',op['analysis'])]
  fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
  for ax,(label,an) in zip(axes,phase_groups):
   phases=an['iterations'];names=[p['name'].split('_phase_')[-1] for p in phases];active=[p['gpu_activity_ms']/1000 for p in phases];idle=[p['no_gpu_activity_ms']/1000 for p in phases];p=np.arange(len(phases));ax.barh(p,active,label='GPU 事件区间并集',color='#087F76');ax.barh(p,idle,left=active,label='无记录 GPU 活动',color='#D7E0E7');ax.set(yticks=p,yticklabels=names,xlabel='插桩阶段 wall time / s');ax.invert_yaxis();ax.set_title(label,loc='left');ax.grid(axis='x');ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),fontsize=9)
  fig.savefig(figdir/'pipeline.svg');plt.close(fig)
 plots()
 standard_rows=''.join(f'<tr><td>{escape(k[3:])}</td><td>{r["residue_lengths"][0]}</td><td>{r["latency"]["median"]*1000:.3f}</td><td>{r["latency"]["p90"]*1000:.3f}</td><td>{r["latency"]["cv_percent"]:.2f}%</td><td>{r["throughput"]["proteins_per_second"]:.2f}</td><td>{r["memory"]["allocated_peak_gib"]:.3f}</td><td>{link(root/"esm3_standard"/k/"report.json","报告")}</td></tr>' for k,r in standard)
 task_rows=''
 for name in cases:
  r=cases[name]['report'];q=r['quality'];detail=f'CA RMSD {q["ca_rmsd_angstrom"]:.3f} Å / {q["rmsd_matched_residues"]}位置' if 'ca_rmsd_angstrom' in q else (f'掩码恢复 {q["masked_recovery_fraction"]*100:.1f}%' if 'masked_recovery_fraction' in q else '完整有效张量有限')
  setting=f'{r["task"]} / B{r["config"]["batch_size"]} / L{r["residue_lengths"]} / {r["config"]["precision"]} / {r["config"]["device"]}'
  allocated=r['memory']['allocated_peak_gib'];memory_display=f'{allocated:.3f}' if allocated is not None else '—（CPU）'
  task_rows+=f'<tr><td>{escape(name)}<br><small>{escape(setting)}</small></td><td>{r["latency"]["median"]:.4f}</td><td>{r["end_to_end"]["median"]:.4f}</td><td>{memory_display}</td><td>{escape(detail)}</td><td>{link(root/"esm3"/name/"report.json","报告")}</td></tr>'
 dmsrows=''.join(f'<tr><td>{escape(name)}<br><small>masked-forward / FP32 / L{r["target_residues"]}<br>实际context B≤{min(fitness["parameters"]["batch_size"],r["masked_contexts"])} / {r["latency_seconds"]["count"]}次sweep</small></td><td>{r["rows"]:,}</td><td>{r["masked_contexts"]}</td><td>{r["quality"]["spearman"]:.3f}</td><td>{r["quality"]["auc"]:.3f}</td><td>{r["latency_seconds"]["median"]:.3f}</td><td>{r["masked_contexts_per_second"]:.1f}</td><td>{link(root/"esm3_fitness"/name/"predictions.csv","全部评分")}</td></tr>' for name,r in fitness['assays'].items())
 native_rows=''
 for name,an,path in systems:
  top=an['kernels'][0];native_rows+=f'<tr><td>{escape(name)}</td><td>{an["kernel_count"]:,}</td><td>{an["gpu_activity_fraction_percent"]:.1f}%</td><td class="kernel">{escape(top["name"])}</td><td>{top["kernel_sum_percent"]:.1f}%</td><td>{link(path,".nsys-rep")}</td></tr>'
 comp=read(root/'openfold_cpu_comparison.json') if (root/'openfold_cpu_comparison.json').exists() else None
 batch_fail=sum(c['status']!='pass' for c in panel['batch_consistency'].values());gains=''
 om=quality['metrics'];olat=of['latency_seconds'];ot=of['output'];count=sum(r['rows'] for r in fitness['assays'].values())
 gemm=sum(k['kernel_sum_percent'] for k in ons['analysis']['kernels'] if any(s in k['name'].lower() for s in ['gemm','gemv']))
 micro=read(root/'operator_diagnostics/microbench.json')
 catalog=build_catalog(root,esm,panel,fitness,batch,of,ns,ons,op,micro,read,sources)
 write_results_guide(catalog,root)
 for p in [root/'experiment_index.json',root/'RESULTS.md',REPO/'scripts/evaluation_catalog.py']:
  sources[str(p)]=sha256(p)
 overview=render_overview(catalog,esm,panel,fitness,of,len(systems),link,root)
 setting_index=render_catalog(catalog,root,link)
 native_settings={str(Path(r['evidence'][-1]).resolve()):r for r in catalog['records'] if r['group']=='systems'}
 native_rows=''
 for name,an,path in systems:
  top=an['kernels'][0];r=native_settings[str(path.resolve())]
  setting=f'{r["task"]} / B{r["batch_size"]} / L{r["residue_lengths"]} / {r["precision"]}'
  capture=f'{r["settings"]["capture_scope"]}；预热{r["warmup"]} / 捕获{r["runs"]}'
  native_rows+=f'<tr><td>{escape(name)}<br><small>{escape(setting)}<br>{escape(capture)}</small></td><td>{an["kernel_count"]:,}</td><td>{an["gpu_activity_fraction_percent"]:.1f}%</td><td class="kernel">{escape(top["name"])}</td><td>{top["kernel_sum_percent"]:.1f}%</td><td>{link(path,".nsys-rep")}<br>{link(Path(r["source"]),"settings")}</td></tr>'
 folder_roles=[('esm3/','核心任务、长度边界、B2和CPU参考','case/report.json、tensors.pt、operators.json'),
 ('esm3_standard/','十target B1性能与真实混合B4','panel_summary.json、case/report.json'),
 ('esm3_batch_fp32/','FP32的padding与B1/B4数值诊断','batch_diagnostic.json'),
 ('esm3_fitness/','完整三个DMS实验，逐assay存储','assay/predictions.csv、score_tables.pt'),
 ('openfold/','完整单体参考预测、结构质量与模块热点','report.json、features.pt、tensors.pt、prediction_unrelaxed.pdb'),
 ('esm3/nsight/、esm3_standard/nsight/','ESM3的warm/pipeline原生capture','*.nsys-rep、*.sqlite、*_workload/workload.json'),
 ('openfold_nsight/、openfold_pipeline/','OpenFold的warm/pipeline原生capture','*.nsys-rep、workload/workload.json、nsight_summary.json'),
 ('operator_diagnostics/','模型shape驱动的简化算子诊断','microbench.json'),
 ('figures/','从上面实测数据导出的图表','SVG / PNG；不是新的实验')]
 layout_rows=''.join(f'<tr><td><code>{escape(p)}</code></td><td>{escape(role)}</td><td><code>{escape(files)}</code></td></tr>' for p,role,files in folder_roles)
 layout_html=f'''<section id="layout"><h2>结果目录 / 按模型、实验族、证据类型查找</h2><p>统一根目录：<code>{escape(root.name)}/</code>。先看 <code>report.html</code>，再按 <code>experiment_index.json</code> 找实验；<code>metrics.csv</code>只汇总正式性能。<code>execution.json</code>记录执行阶段，<code>verification.json</code>检查交付完整性，两者都不是“科学质量全部通过”的标记。</p><div class="scroll"><table class="directory-table"><tr><th>目录层级</th><th>实验职责</th><th>关键文件</th></tr>{layout_rows}</table></div><p>例：<code>esm3/L256/</code>表示合成序列 forward / B1 / L256 / BF16；<code>esm3/structure/</code>表示泛素 B1 / L76 / 8步结构生成；<code>esm3/crambin_structure/</code>为 L46。<code>mixed_B2</code>与<code>mixed_B4</code>是两个不同输入集合。索引为每个短名补全任务、B、L、精度和设备。</p><p><code>warm</code>表示预热后的操作捕获；<code>pipeline</code>表示加载到交付的阶段捕获。保存的原报告、argv和原生文件相互引用这些路径，因此保留历史路径；新实验使用新的顶层输出目录。{link(root/'RESULTS.md','完整目录导航与全部实验表')}</p></section>'''
 coverage_html='''<div class="scroll"><table class="coverage-matrix"><tr><th>评测维度</th><th>本轮实际覆盖</th><th>不能外推的边界</th></tr>
 <tr><td>输入 / 任务 / 长度</td><td>ESM3四核心路径；两小蛋白；合成L256/1024/2048；十个标准target</td><td>ESM3功能注释/任意设计场景未测；OpenFold只实测6KWC L191</td></tr>
 <tr><td>batch / 精度 / 正确性</td><td>ESM3 forward B1/B2/B4；CPU-FP32；BF16批次失败保留；FP32 padding隔离</td><td>生成和OpenFold仅B1；未扫OpenFold长度/批次/精度；阈值不代表MUSA最终验收</td></tr>
 <tr><td>性能 / 容量 / 波动</td><td>加载、首调用、稳态原始样本、median/P90/CV、吞吐、allocated/reserved峰值；输入准备与交付分开</td><td>当前测试规模的显存结果不等于最大容量/OOM边界或并发服务压测；电功率采样不是每请求能耗</td></tr>
 <tr><td>蛋白任务质量</td><td>掩码恢复、两个小结构；三个完整DMS；OpenFold实验结构RMSD/lDDT与CPU坐标</td><td>非217-assay总榜；非大规模独立结构集；未证明预训练数据隔离</td></tr>
 <tr><td>系统 / 算子 / kernel</td><td>真实ATen、shape/dtype/stride；OpenFold三角/OPM/IPA；warm/pipeline原生Systems；8个简化算子</td><td>不是所有任务/长度都做native capture；DMS未单独抓Systems；Compute硬件计数器缺失；CPU采样关闭</td></tr>
 <tr><td>硬件与未来迁移</td><td>当前4090执行，冻结输入/权重/配置；明确后端能力与失败</td><td>A100/S5000、多卡、MUSA部署与40%优化尚未实测，本轮不宣称完成</td></tr>
 </table></div>'''
 micro_rows=''.join(f'<tr><td>{escape(v["operator"])}</td><td>{v["latency"]["median"]*1000:.4f}</td><td>{v["fp32_reference"]["mean_abs_error"]:.4g}</td><td>{escape(str(v["input_shapes"]))}</td></tr>' for v in micro['results'])
 css='''*{box-sizing:border-box}body{margin:0;background:#eff3f6;color:#223747;font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1240px;margin:32px auto;padding:42px 52px;background:white;border:1px solid #d9e3eb}header{border-top:5px solid #087F76;border-bottom:1px solid #d3e0e8;padding:22px 0}h1{font-size:34px;line-height:1.4}h2{margin-top:50px;font-size:25px}h3{font-size:19px}.eyebrow{font-size:13px;color:#087F76;letter-spacing:2px}a{color:#087F76;text-underline-offset:4px}.reading{border-left:4px solid #087F76;background:#eff8f5;padding:16px 21px}.pending{border-color:#A35E20;background:#fff7ed}nav{display:flex;flex-wrap:wrap;gap:18px;padding:18px 0;font-size:14px}.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:24px 0}.facts>div{border-top:1px solid #dbe5ec;padding-top:14px}.facts strong{display:block;font-size:30px;color:#087F76}figure{margin:28px 0;padding:18px 12px;border:1px solid #e0e7ed}img{width:100%;height:auto}figcaption,small,.muted{font-size:14px;color:#617586}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:11px 10px;border-bottom:1px solid #dce5eb;white-space:nowrap}th{background:#f3f6f8}.kernel{font:12px/1.5 monospace;white-space:normal;overflow-wrap:anywhere;max-width:650px}details{padding:14px 18px;border:1px solid #dce5eb;margin:15px 0}summary{cursor:pointer;font-weight:600}code,pre{font:13px/1.8 ui-monospace,monospace}pre{overflow:auto;padding:20px;background:#f3f6f8}footer{border-top:1px solid #dce5eb;margin-top:40px;padding-top:20px;font-size:13px;color:#617586}@media(max-width:720px){main{margin:0;padding:23px 16px}h1{font-size:27px}.facts{grid-template-columns:1fr}figure{padding:6px}}'''

 css+='''.experiment-map{margin:25px 0;border:1px solid #dce5eb;border-radius:10px;overflow:hidden}.map-level{display:grid;grid-template-columns:150px 1fr;gap:18px;padding:22px;background:#f7fafb;border-bottom:1px solid #dce5eb}.map-level:last-child{border:0}.level-mark{font:600 13px/1.7 system-ui;color:#557082;padding-top:5px}.map-shared{border-left:4px solid #087F76;padding:0 18px}.map-models,.map-experiments{display:grid;grid-template-columns:2.6fr 1fr;gap:18px}.model-title{padding:10px 15px;background:#087F76;color:white;font-weight:650;border-radius:5px}.model-title:last-child{background:#365D9A}.map-experiments{position:relative}.esm-branches{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.map-cell{background:white;border:1px solid #dbe5ec;border-top:3px solid #087F76;border-radius:6px;padding:15px;font-size:13px;line-height:1.8}.map-experiments>.map-cell{border-top-color:#365D9A}.map-cell h4{margin:0 0 12px;font-size:15px}.map-cell p{margin:8px 0 0}.map-evidence{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.map-shared p{margin:8px 0;font-size:14px}.setting-glossary{display:grid;grid-template-columns:1fr 1fr;gap:0 25px;padding:12px 20px;background:#f5f8fa;font-size:14px}.settings-table td{white-space:normal;vertical-align:top;min-width:125px}.settings-table td:first-child{max-width:235px;overflow-wrap:anywhere}.settings-table td:last-child{min-width:380px}.settings-table pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:620px}.catalog-group>summary{font-size:15px}.directory-table td{white-space:normal;vertical-align:top}.directory-table code{overflow-wrap:anywhere}.coverage-matrix td{white-space:normal;vertical-align:top}@media(max-width:1000px){.map-level{grid-template-columns:1fr}.map-models,.map-experiments{grid-template-columns:1fr}.esm-branches{grid-template-columns:1fr}.map-evidence{grid-template-columns:1fr}.setting-glossary{grid-template-columns:1fr}}'''
 content=f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ESM3 / OpenFold · 4090 原版评测</title><style>{css}</style></head><body><main><header><p class="eyebrow">REFERENCE EVALUATION / RTX 4090 / 2026-10-09</p><h1>ESM3 与 OpenFold：4090 评测结果</h1><p>原版模型 · 正确性与性能分开 · 原生时间线 · 每张图都有定性解读</p></header><nav><a href="#overview">全局概览</a><a href="#settings">实验设置</a><a href="#layout">结果目录</a><a href="#esm">ESM3性能</a><a href="#correct">正确性</a><a href="#quality">功能评分</a><a href="#of">OpenFold</a><a href="#nsight">热点</a><a href="#pipeline">阶段时间线</a><a href="#scope">范围与复用</a></nav><p class="reading">本轮只完成参考评测。已移除优化模块与相关结果，完整计算原版几何路径；不展示优化对照、40%加速结论或未拥有硬件的模拟结果。</p><div class="facts"><div>ESM3核心任务<strong>4 条路径</strong>两真实小蛋白 / 合成长度边界 / 标准数据</div><div>OpenFold完整预测<strong>{olat['median']:.3f} s</strong>191残基 / FP32 / 4遍主干</div><div>原生 Systems 报告<strong>{len(systems)} 份</strong>Compute计数器权限受限，指标缺失</div></div>
 {overview}
 {setting_index}
 <h2 id="esm">01 / ESM3：实际长度、延迟与显存</h2><p>固定 esm 3.2.3、Torch 2.3.1+cu121、开放权重 revision；BF16 / B1 / seed7，3次预热后20次正式同步测量。标准性能输入来自固定 ProteinGym v1.3，按长度桶和分类元数据选10个不同target，未用预测或实验分数挑选。部分target是结构域/工程蛋白，不一概称完整天然序列。</p>
 {image('lengths','10个真实target的稳态延迟和显存；未画拟合线。allocated是PyTorch实际占用峰值，不是整卡nvidia-smi显存。')}
 <p class="reading">长输入的延迟和显存增加明显，说明短蛋白跑通不能说明长蛋白容量。同图序列身份与长度都变化，不能单凭这些点证明严格计算复杂度。L2048合成边界另有独立原版运行；未修改batch/长度来隐藏OOM。</p><div class="scroll"><table><tr><th>target（forward / B1 / BF16）</th><th>残基 L</th><th>median/ms</th><th>P90/ms</th><th>CV</th><th>proteins/s</th><th>allocated/GiB</th><th>原始</th></tr>{standard_rows}</table></div>
 {image('task_timing','ESM3首调用、稳态操作和热交付分别记录；forward为L256，生成任务为泛素L76；均B1、主干BF16。sequence/inverse生成3步，structure生成8步。对数轴展示跨量级时间，不能按视觉距离计算绝对差。')}
 <p class="reading">首调用通常包含首次kernel/辅助模块准备；热交付增加输入准备、CPU输出和PDB生成。模型加载另记录，不属于稳态或热交付。一次最快调用不能代表服务性能，median/P90/CV和全部原始样本应同时看。</p><div class="scroll"><table><tr><th>任务与 settings</th><th>稳态/s</th><th>热交付/s</th><th>allocated/GiB</th><th>功能/质量</th><th>原始</th></tr>{task_rows}</table></div><p>已知位置保留、掩码补全、有限坐标是功能检查；单个种子三个掩码位的恢复率不能作为蛋白设计总体准确率。结构RMSD先做Cα刚体对齐，消除旋转和平移；这里只有1UBQ和1CRN两个小结构。sequence/inverse遮住第4、5、13位、生成3步；structure只给序列、生成8步，实验PDB仅用于事后比较；温度0.7、linear调度、entropy策略，seed7。CPU行是独立FP32数值参考，不能与BF16生成任务混为同配置性能比较。</p>
 <h2 id="correct">02 / 数值、真实batch与失败保留</h2><p>CPU-FP32与CUDA-FP32比较同token/mask和全部有效logits/embeddings；B1 / L76 / CPU线程4 / seed7。结果 {esm['comparisons']['cpu_cuda_fp32']['status']}。比较不把padding、BOS/EOS计入有效残基，也不能只看argmax或余弦。</p>
 {image('batch_errors','标准数据四个target的L=47/237/413/852，B1与padding到宽854的B4比较；每种精度内部比较。relative L2为误差范数/参考范数，对数轴表示量级。')}
 <p class="reading pending">BF16的{batch_fail}/4个完整输出检查超出预设atol=.1、rtol=.02，标准性能面板状态为{panel['status']}，没有事后放宽阈值。FP32单独诊断状态为{batch['status']}：对照补齐长度、batch分组，并替换padding内容检验隔离；有效输出不随padding字母改变。结果支持低精度/计算顺序差异的解释，不能把BF16结果自动改为通过。</p><p>{link(root/'esm3_standard/panel_summary.json','BF16全部数值误差')} · {link(root/'esm3_batch_fp32/batch_diagnostic.json','FP32隔离诊断和阈值')} · {link(root/'esm3/audit_summary.json','CPU对照与全部配置')}</p><p>forward支持真B2/B4与变长padding。生成当前明确B1；OpenFold此配置每次处理一个单体请求。未将逐请求循环称为模型并行batch；未来批处理支持可在此流程新增独立配置验证。</p>
 <h2 id="quality">03 / 功能质量：三个完整DMS实验</h2><p>共{count:,}变体，采用WT单位置masked-marginal：遮住实验测过的位置，分数为突变氨基酸log概率减WT的log概率。相同位置归一化项抵消，等价于logits相减。多突变求和，不能直接刻画联合epistasis。固定FP32、最多B8、完整target context，不用标签拟合或调参。</p>
 {image('fitness','所有变体参与40×40二维排名直方图，每格除以本实验样本总数；共用对数色标，空格留白。虚线是排名同向，不是拟合线。')}
 <p class="reading">BLAT的排名沿左下—右上集中，排序相关约.757；RL40A约.286、SPG1约.106，当前方法的整体排序较弱。加性多突变的表达限制是可能原因之一，不能仅凭这张图证明唯一原因。AUC按官方二值标签评价，与连续fitness的Spearman不同；不合并成未运行的217实验总榜。</p><div class="scroll"><table><tr><th>assay</th><th>变体</th><th>masked上下文</th><th>Spearman</th><th>AUC</th><th>sweep/s</th><th>contexts/s</th><th>原始</th></tr>{dmsrows}</table></div><details><summary>为何155,396变体不等于155,396次模型推理？</summary><p>变体复用每个WT masked位置的预测，SPG1的149,360行只需要4个位置的分数表。计时是覆盖全部位置的一次sweep，包括gather和小表CPU传输，排除CSV解析、查表、排名和落盘。不能用变体数除以模型耗时伪造完整变体推理吞吐。</p><p>三预定位置的B1/B8全64logits检查通过；每一行WT字母、位置和mutated_sequence均核验。工程PASS不等于科学质量达标，未设事后准确率通过线；没有用标签训练也不证明与预训练数据无重叠。</p></details>
 <h2 id="of">04 / OpenFold：完整单体模型参考</h2><p>独立Python3.10/Torch2.5.1+cu124环境，固定官方commit与公开权重，真实6KWC预计算MSA。48个Evoformer blocks、3次recycling（4遍主干）及结构头完整运行；FP32、TF32关闭、chunk=64且不自动调优。这是公开评测参数，不是模型优化。B1 / L191 / seed42 / CPU线程8；MSA特征shape=[512,191,49,4]、ExtraMSA=[5120,191,4]，末维4代表recycling遍数，不代表batch；padding后的MSA行数不等于有效比对序列数。</p><p>加载 {of['phases_seconds']['model_load']:.3f}s，准备 {of['phases_seconds']['prepare']:.3f}s，首预测 {of['phases_seconds']['first_inference']:.3f}s；{olat['count']}次正式模型调用 median {olat['median']:.3f}s、P90 {olat['p90']:.3f}s、CV {olat['cv_percent']:.2f}%，allocated peak {of['memory']['allocated_peak_gib']:.3f}GiB。加载/准备/CPU交付、PDB与profiler排除在稳态计时之外。</p><p class="reading">对实验6KWC A链的{quality['alignment']['valid_ca_pairs']}个有效同字母Cα，覆盖{quality['alignment']['prediction_ca_coverage']*100:.1f}%；Kabsch RMSD {om['ca_rmsd_angstrom']:.3f}Å、Cα lDDT {om['ca_lddt']:.4f}。低RMSD表示整体对齐偏差小，高lDDT表示局部距离保持好；平均pLDDT {ot['plddt_mean']:.2f}只表示模型置信度。单官方示例不是标准结构数据集总体精度或已证明独立测试。</p><p>{link(root/'openfold/report.json','完整配置与计时')} · {link(root/'openfold/prediction_unrelaxed.pdb','未松弛PDB')} · {link(root/'openfold/structure_quality.json','结构映射和质量')} · {link(root/'openfold/protein_modules.json','实际三角/OPM/IPA模块shape')} · {link(root/'openfold/operators.json','完整ATen热点')}</p>
 '''
 if comp:
  content+=f'<p class="reading">同一冻结特征的CPU-FP32/CUDA-FP32比较：{comp["status"]}，有效坐标MAE {comp["metrics"]["final_atom_positions"]["mae"]:.6g}Å，Cα对齐RMSD {comp["ca_aligned_rmsd_angstrom"]:.6g}Å。CPU选同版本官方torch attention/IPA分支，CUDA用原扩展；输入、权重、配置、atom mask匹配，实现差异明确。CPU复用已完成完整参考，未重新宣称CPU性能排名。{link(root/"openfold_cpu_comparison.json","完整对照与阈值")}</p>'
 content+=f'''<p class="reading pending">实测边界：预计算MSA、无模板、未松弛单体；没有运行MSA数据库搜索、模板搜索、OpenMM relaxation或多体模型。这里“完整”指此冻结网络配置完整执行，不能写成所有OpenFold分支都已测试。</p>
 <h2 id="nsight">05 / 原生热点：从模型算子到实际kernel</h2>{image('hotspots','每行按该捕获全部kernel累计时间归一化；K1/K2/K3是该行的排名，不表示各行同一kernel。其余保留，不截断后重新归一化。')}
 <p class="reading">ESM3任务和OpenFold的热点分布不同，不能用一个短序列forward的GEMM代表整条蛋白流水线。OpenFold GEMM/GEMV名称族合计约{gemm:.1f}%；三角/OPM/IPA的语义由实际模块hooks补充，ATen GEMM名字本身不能证明具体蛋白模块。</p><p>{link(root/'operator_coverage.json','任务书算子覆盖：实际证据与未观察项')}</p><div class="scroll"><table><tr><th>捕获与 settings</th><th>launch</th><th>GPU activity</th><th>K1完整名字</th><th>K1占比</th><th>原生</th></tr>{native_rows}</table></div>
 <p class="reading pending">GPU activity是捕获区间GPU事件并集占比，不等于occupancy。kernel累计占比也不等于wall时间占比。当前RmProfilingAdminOnly=1且无CAP_SYS_ADMIN；Compute已计划实际name/grid/block/registers变体，计数器保持null。没有真实DRAM/cache/occupancy，不能声称compute-bound、memory-bound或实测Roofline。CPU采样/上下文切换关闭，NVTX/CUDA/OSRT仍保留。</p><p>{link(root/'esm3/nsight/nsight_summary.json','ESM3时间线/Compute目标')} · {link(root/'openfold_nsight/nsight_summary.json','OpenFold权限/Compute目标')}</p><details><summary>8个shape驱动算子小基准</summary><p>BF16 / 输入seed7 / 每算子预热5次、正式30次；B和L不能代替各tensor的shape。输入由CPU固定种子生成，采用模型中实测shape及注意力head维度；这是简化诊断，不是实际激活/权重/mask重放。LayerNorm未包含完整仿射，SDPA是无mask路径。误差对照CPU-FP32的同一量化输入，不作为整任务容差。小算子时间不能当作模型加速或实测Roofline。</p><div class="scroll"><table><tr><th>算子</th><th>同步median/ms</th><th>对FP32的MAE</th><th>输入shape</th></tr>{micro_rows}</table></div><p>{link(root/'operator_diagnostics/microbench.json','原始时间、stride、dtype、输入hash及简化限制')}</p></details>
 <h2 id="pipeline">06 / 将热点放回完整阶段</h2>{image('pipeline','左：ESM3 inverse / B1 / L76 / BF16 / 3步；右：OpenFold structure / B1 / L191 / FP32 / 4遍主干。新进程首次加载的插桩诊断：加载、输入准备、模型预测、CPU/PDB交付；GPU活动取事件区间并集，不重复累加重叠。横轴是插桩阶段时间。')}
 <p class="reading">加载与准备可能占新进程交付的明显比例，而warm kernel捕获看不到这些阶段。无记录GPU活动的灰段只说明该区间未看到GPU事件，不能自动解释为Python慢、磁盘慢或CPU瓶颈；原因需要相关IO/CPU证据。OpenFold准备包含真实MSA特征转换，仍不包含外部数据库搜索。插桩时长仅诊断，不替代前述正式benchmark。</p>
 <h2 id="scope">07 / 当前覆盖与以后换卡复用</h2>{coverage_html}<p>两模型评测覆盖环境/权重/输入溯源，原始时间和显存，完整有效输出、实验结构/功能质量，实际算子shape/dtype/stride，warm/pipeline原生时间线及Compute采集流程。结果中的失败和缺失值也是结论，不通过删除失败、减小工作负载或填零把它们写成全PASS。</p><p>任务书的MUSA部署、40%优化和多卡通信属于后续阶段；本轮按用户要求不执行。A100沿用同CUDA评测契约但重建对应架构扩展、重测实际kernel；S5000复用输入/权重/参数/统计/数值比较，设备API、扩展和原生profiler必须实际适配。当前未声称S5000兼容，也不把请求MUSA移到CPU。</p><p>{link(root/'metrics.csv','两模型正式指标CSV')} · {link(REPO/'docs/evaluation_scope.md','任务书覆盖矩阵')} · {link(REPO/'docs/gpu_optimization/openfold_reference.md','OpenFold运行契约')} · {link(REPO/'docs/gpu_optimization/proteingym_reference.md','标准数据评分契约')} · {link(REPO/'README.md','复跑入口')}</p><pre>python scripts/run_evaluation.py --output outputs/新的评测目录
python scripts/build_evaluation_report.py --output outputs/新的评测目录</pre><p>两独立Python环境和权重路径可通过参数指定；默认生成B1与OpenFold单体边界保留。所有输出目录保护已有结果。</p>{layout_html}<footer>{link(root/'report_provenance.json','数据与图表hash')} · {link(root/'cleanup_manifest.json','优化结果清理清单')} · {link(root/'verification.json','交付核验')}。图表均来自保存的实测，不伪造硬件计数器或跨卡结果。</footer></main></body></html>'''
 (root/'report.html').write_text(content,encoding='utf-8');write_json(root/'report_provenance.json',{'sources':sources,'renderer_sha256':sha256(__file__),'html_sha256':sha256(root/'report.html'),'figures':{p.name:sha256(p) for p in figdir.iterdir()}});print(root/'report.html')


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);main(p.parse_args())
