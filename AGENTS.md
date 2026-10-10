# Protein Model Testing
默认中文。当前阶段只评测 ESM3/OpenFold 的原版参考实现，不运行优化收益实验。
代码和结果来源以保存的 source/weight/input SHA-256 为准。CUDA 不代表 MUSA 已适配。
正式计时、数值检查、任务质量和 profiler 分开。保留 BF16 batch 失败与 Compute 权限缺失。
ESM3/OpenFold 使用各自环境，解释器和权重路径显式传入。新实验写 outputs/ 下新目录。
公开基线在 results/rtx4090/evaluation_20261009_01/；大型证据从 Release 下载。
不要修改历史原始指标或把模型权重、账户凭据、环境目录提交到 Git。
