# 当前阶段：仅评测（2026-10-09）

用户明确将工作限定为 ESM3 / OpenFold 在当前4090上的完整参考评测，并保留将来换卡可复用的流程。不开发、运行或保留优化代码，不以40%加速或未拥有的硬件作为本轮终点。

| 任务书内容 | 此阶段对应评测 | 边界 |
|---|---|---|
| ESM3核心任务 | forward、序列补全、序列到结构、结构条件补全；两真实小蛋白 | 不是所有功能注释/设计场景总体质量 |
| OpenFold结构预测 | 固定公开权重和真实MSA，完整网络/recycling，输出与实验结构/CPU比较 | 单体无模板、预计算MSA、未松弛；搜索/模板/relax未测 |
| 算子分析 | ATen实际时间、shape/dtype/stride，几何/三角/OPM/IPA模块证据 | 未执行不等于不支持；Diffusion/Graph不冒充已观察 |
| 性能 | 首调用/稳态/热交付、吞吐、显存、原始分布、监控 | 同步计时；profiler排除；电功率采样不等于每请求能耗 |
| 标准数据 | ProteinGym v1.3十target性能、三个完整DMS实验功能评分 | 不是217实验总榜或已验证训练隔离；结构仅有限示例 |
| 正确性 | 全有效张量、CPU对照、原生回放、残基覆盖/结构误差 | BF16变长batch超阈值如实报告；不放宽容差以换PASS |
| batch | forward真实B2/B4、变长/padding、显存/OOM | ESM生成B1；OpenFold按请求完整预测，未证明模型并行batch |
| Nsight Systems | 各核心任务warm/pipeline、OpenFoldwarm/pipeline，原生报告 | CPU采样/上下文切换关闭，OSRT/NVTX/CUDA保留 |
| Nsight Compute | 实际kernel变体与launch计划、权限预检、可用时采集和launch核验 | 当前宿主计数器权限不足；无真实occupancy/DRAM/cache/roofline |
| 统一后端与未来换卡 | Backend同步/Event/显存能力、显式dtype、冻结输入/权重 | A100/S5000未实测，OpenFold MUSA未移植就明确拒绝 |
| 40%性能提升、MUSA部署、多卡通信 | 后续项目阶段 | 不进入此次评测代码、结果或完成判断 |

本阶段“流程已交付”与“每个数值检查通过”分开：流程必须保留并解释失败，不能声称所有配置均正确或原PDF完整毕设已验收。
