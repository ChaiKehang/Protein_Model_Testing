# 原版评测契约

本阶段按用户2026-10-09指令只评测4090上的ESM3和OpenFold。范围矩阵以 `evaluation_scope.md` 为准；不运行或保留模型优化分支。

冻结代码、权重revision/SHA、输入序列/MSA/特征hash、任务、逻辑device、精度、autocast/TF32、attention、chunk、seed、真实batch和计时边界。脚本保护已有输出目录；失败/OOM和原始时间必须保留。

计时区分模型加载、输入准备、首调用、稳态操作、热交付；GPU操作完成再读wall时钟，device Event另存。报告全部样本与median/P90/CV；proteins/s按实际完成蛋白数除以累计同步时间，残基吞吐排除padding。profiling另跑，不参与正式统计。NVML监控覆盖整进程，不据粗粒度采样声称每次推理能耗。

数值比较首先核验输入token/mask/特征及权重配置；比较全部有效logits/embeddings和结构有效原子，不能只用形状、argmax、余弦或单案例pLDDT。诊断阈值在运行前声明，超出即如实报告。结构质量先记录映射/覆盖，再做刚体对齐与Cα lDDT；设计/功能科学质量与工程运行成功分开。

ESM3生成当前B1；forward可真batch/padding。OpenFold是预计算MSA、无模板、未松弛完整单体模型。两模型范围、不同计时API与置信度尺度不得混合比较。

未来换卡复用契约并重新测设备/依赖/算子/误差。请求MUSA但未适配时明确失败，不静默CPU回退。A100/S5000和多卡不是当前实验对象。
