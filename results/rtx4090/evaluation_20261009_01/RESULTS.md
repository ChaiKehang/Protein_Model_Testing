# 4090 评测结果导航

先打开 [report.html](report.html)；逐实验设置见 [experiment_index.json](experiment_index.json)，正式性能指标见 [metrics.csv](metrics.csv)。

## 层级与文件职责

| 路径 | 内容 |
|---|---|
| `esm3/` | 核心任务、合成长度边界、B2、CPU对照；每个case独立目录 |
| `esm3_standard/` | 十个标准 target 的B1与混合B4；panel_summary保留BF16比较失败 |
| `esm3_batch_fp32/` | B1 / padding B1 / B4 / padding内容隔离的FP32诊断 |
| `esm3_fitness/` | 三个独立assay；predictions.csv是逐变体评分，score_tables.pt是复用的上下文输出 |
| `openfold/` | 正式预测、features、完整输出、PDB、结构质量、ATen与蛋白模块 |
| `esm3/nsight/、esm3_standard/nsight/` | ESM3原生warm/pipeline捕获；与正式报告相互引用 |
| `openfold_nsight/、openfold_pipeline/` | OpenFold原生warm与pipeline；Compute权限与计划保存在summary |
| `operator_diagnostics/` | 简化算子诊断，不等同真实模型激活重放 |
| `figures/` | 由实测数据生成的可视化；来源hash在report_provenance.json |

## 命名规则与历史短名

`L256` = ESM3 forward / 合成random_L256 / B1 / BF16；`sequence`、`structure`、`inverse` = 泛素76残基 / B1；`crambin_*` = 1CRN46残基。
`mixed_B2` = 胰岛素A链21 + 泛素76；`mixed_B4` = 标准target47/237/413/852。它们均是真forward批次，生成不使用B2/B4。
`*_warm` = 预热后操作范围；`*_pipeline` = 新进程加载/准备/首次任务/交付范围。相同名字位于不同目录时，必须携带完整相对路径引用。
原始目录与文件路径已被原生报告、记录的argv及hash引用，因此保留这些路径；索引补足其含义，不篡改历史报告。

## 全部实验

| ID | 任务 | B | L/残基 | 精度 | 原始证据 |
|---|---|---|---|---|---|
| `esm3/L256` | forward | 1 | [256] | bf16 | [report.json](esm3/L256/report.json) |
| `esm3/L1024` | forward | 1 | [1024] | bf16 | [report.json](esm3/L1024/report.json) |
| `esm3/L2048` | forward | 1 | [2048] | bf16 | [report.json](esm3/L2048/report.json) |
| `esm3/mixed_B2` | forward | 2 | [21, 76] | bf16 | [report.json](esm3/mixed_B2/report.json) |
| `esm3/sequence` | sequence | 1 | [76] | bf16 | [report.json](esm3/sequence/report.json) |
| `esm3/structure` | structure | 1 | [76] | bf16 | [report.json](esm3/structure/report.json) |
| `esm3/inverse` | inverse | 1 | [76] | bf16 | [report.json](esm3/inverse/report.json) |
| `esm3/ubiquitin_fp32` | forward | 1 | [76] | fp32 | [report.json](esm3/ubiquitin_fp32/report.json) |
| `esm3/ubiquitin_cpu_fp32` | forward | 1 | [76] | fp32 | [report.json](esm3/ubiquitin_cpu_fp32/report.json) |
| `esm3/crambin_forward` | forward | 1 | [46] | bf16 | [report.json](esm3/crambin_forward/report.json) |
| `esm3/crambin_sequence` | sequence | 1 | [46] | bf16 | [report.json](esm3/crambin_sequence/report.json) |
| `esm3/crambin_structure` | structure | 1 | [46] | bf16 | [report.json](esm3/crambin_structure/report.json) |
| `esm3/crambin_inverse` | inverse | 1 | [46] | bf16 | [report.json](esm3/crambin_inverse/report.json) |
| `esm3_standard/pg_AMFR_HUMAN` | forward | 1 | [47] | bf16 | [report.json](esm3_standard/pg_AMFR_HUMAN/report.json) |
| `esm3_standard/pg_BBC1_YEAST` | forward | 1 | [64] | bf16 | [report.json](esm3_standard/pg_BBC1_YEAST/report.json) |
| `esm3_standard/pg_A0A1I9GEU1_NEIME` | forward | 1 | [161] | bf16 | [report.json](esm3_standard/pg_A0A1I9GEU1_NEIME/report.json) |
| `esm3_standard/pg_RPC1_LAMBD` | forward | 1 | [237] | bf16 | [report.json](esm3_standard/pg_RPC1_LAMBD/report.json) |
| `esm3_standard/pg_ADRB2_HUMAN` | forward | 1 | [413] | bf16 | [report.json](esm3_standard/pg_ADRB2_HUMAN/report.json) |
| `esm3_standard/pg_GCN4_YEAST` | forward | 1 | [281] | bf16 | [report.json](esm3_standard/pg_GCN4_YEAST/report.json) |
| `esm3_standard/pg_Q837P4_ENTFA` | forward | 1 | [589] | bf16 | [report.json](esm3_standard/pg_Q837P4_ENTFA/report.json) |
| `esm3_standard/pg_A0A192B1T2_9HIV1` | forward | 1 | [852] | bf16 | [report.json](esm3_standard/pg_A0A192B1T2_9HIV1/report.json) |
| `esm3_standard/pg_BRCA1_HUMAN` | forward | 1 | [1863] | bf16 | [report.json](esm3_standard/pg_BRCA1_HUMAN/report.json) |
| `esm3_standard/pg_UBE4B_MOUSE` | forward | 1 | [1173] | bf16 | [report.json](esm3_standard/pg_UBE4B_MOUSE/report.json) |
| `esm3_standard/mixed_B4` | forward | 4 | [47, 237, 413, 852] | bf16 | [report.json](esm3_standard/mixed_B4/report.json) |
| `diagnostic/esm3_cpu_cuda` | forward | 1 | [76] | fp32 | [audit_summary.json](esm3/audit_summary.json) |
| `diagnostic/fp32_padding_batch` | forward | [1, 4] | [47, 237, 413, 852] | fp32 | [batch_diagnostic.json](esm3_batch_fp32/batch_diagnostic.json) |
| `diagnostic/bf16_B1_B4` | forward | [1, 4] | [47, 237, 413, 852] | bf16 | [panel_summary.json](esm3_standard/panel_summary.json) |
| `esm3_fitness/RL40A_YEAST_Mavor_2016` | masked_forward | 8 | [128] | fp32 | [dms_summary.json](esm3_fitness/dms_summary.json) |
| `esm3_fitness/BLAT_ECOLX_Firnberg_2014` | masked_forward | 8 | [286] | fp32 | [dms_summary.json](esm3_fitness/dms_summary.json) |
| `esm3_fitness/SPG1_STRSG_Wu_2016` | masked_forward | 8 | [448] | fp32 | [dms_summary.json](esm3_fitness/dms_summary.json) |
| `openfold/6KWC` | structure | 1 | [191] | fp32 | [report.json](openfold/report.json) |
| `diagnostic/openfold_cpu_cuda` | structure | 1 | [191] | fp32 | [openfold_cpu_comparison.json](openfold_cpu_comparison.json) |
| `operator/linear` | linear | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `operator/layer_norm` | layer_norm | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `operator/softmax` | softmax | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `operator/vector_norm` | vector_norm | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `operator/topk` | topk | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `operator/embedding` | embedding | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `operator/batch_gemm` | batch_gemm | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `operator/sdpa` | sdpa | 见tensor shape | 不按蛋白L定义 | bf16 | [microbench.json](operator_diagnostics/microbench.json) |
| `esm3/nsight/L256_warm` | forward | 1 | [256] | bf16 | [workload.json](esm3/nsight/L256_warm_workload/workload.json) |
| `esm3/nsight/L1024_warm` | forward | 1 | [1024] | bf16 | [workload.json](esm3/nsight/L1024_warm_workload/workload.json) |
| `esm3/nsight/mixed_B2_warm` | forward | 2 | [21, 76] | bf16 | [workload.json](esm3/nsight/mixed_B2_warm_workload/workload.json) |
| `esm3/nsight/sequence_warm` | sequence | 1 | [76] | bf16 | [workload.json](esm3/nsight/sequence_warm_workload/workload.json) |
| `esm3/nsight/structure_warm` | structure | 1 | [76] | bf16 | [workload.json](esm3/nsight/structure_warm_workload/workload.json) |
| `esm3/nsight/inverse_warm` | inverse | 1 | [76] | bf16 | [workload.json](esm3/nsight/inverse_warm_workload/workload.json) |
| `esm3/nsight/L256_pipeline` | forward | 1 | [256] | bf16 | [workload.json](esm3/nsight/L256_pipeline_workload/workload.json) |
| `esm3/nsight/sequence_pipeline` | sequence | 1 | [76] | bf16 | [workload.json](esm3/nsight/sequence_pipeline_workload/workload.json) |
| `esm3/nsight/structure_pipeline` | structure | 1 | [76] | bf16 | [workload.json](esm3/nsight/structure_pipeline_workload/workload.json) |
| `esm3/nsight/inverse_pipeline` | inverse | 1 | [76] | bf16 | [workload.json](esm3/nsight/inverse_pipeline_workload/workload.json) |
| `esm3_standard/nsight/BRCA1_original` | forward | 1 | [1863] | bf16 | [workload.json](esm3_standard/nsight/BRCA1_original_workload/workload.json) |
| `openfold_nsight/6KWC` | structure | 1 | [191] | fp32 | [workload.json](openfold_nsight/workload/workload.json) |
| `openfold_pipeline/6KWC` | structure | 1 | [191] | fp32 | [workload.json](openfold_pipeline/workload/workload.json) |

## 如何判断结果

1. 正式性能只比较相同任务/输入/B/L/精度/计时范围的latency；Systems插桩时长另看。
2. 模型执行PASS不等于数值一致性PASS或功能质量达标；BF16批次失败与Compute缺失保留。
3. OpenFold此次限无模板、预计算MSA、未松弛单体；A100/S5000和未测分支不能据此宣称通过。
4. `verification.json`检查交付完整性；`execution.json`记录实际阶段退出状态；`cleanup_manifest.json`记录已授权清理。
