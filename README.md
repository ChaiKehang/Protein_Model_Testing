# Protein Model Testing

ESM3 / OpenFold 原版 GPU 评测代码与可追溯结果。当前实测硬件：单张 NVIDIA RTX4090；后续 A100、S5000 分目录保存，只有设置一致时比较。

## 当前成果

- [4090结果目录](results/rtx4090/evaluation_20261009_01/README.md)：性能、正确性、任务质量、逐实验settings和可视化。
- [HTML报告](results/rtx4090/evaluation_20261009_01/report.html)：克隆后在浏览器打开；GitHub页面显示源码。
- [完整原始证据 Release](https://github.com/ChaiKehang/Protein_Model_Testing/releases/tag/rtx4090-reference-20261009)：13份原生Systems、完整张量、trace与校验。
- [评测覆盖与边界](docs/evaluation_scope.md)：BF16批次失败、Compute权限不足和未测分支明确保留。

## 仓库结构

```text
main.py / compare_outputs.py  单配置测量与完整输出数值比较
common/                      设备同步/Event/显存、统计、溯源、监控
models/                      官方固定版本模型适配；不修改网络层数或权重
scripts/                     套件、DMS、结构质量、Nsight、报告与跨卡比较
configs/                     精度、任务参数、依赖与权重revision
inputs/                      小型冻结输入与数据来源manifest
docs/                        环境重建、实验契约和迁移说明
results/<gpu>/<run>/          精简结果；大型证据存Release
```

## 复跑与换卡

详见 [复现步骤](docs/REPRODUCE.md)。ESM3和OpenFold使用两个独立环境。权重不入Git，下载固定revision并校验；正式计时与profiler分开。生成仅B1，OpenFold此配置仅单体B1；forward支持真B2/B4。

跨卡先复用输入/权重/任务/精度/批次/计时范围，再检查完整输出，最后比较性能；CUDA运行不证明MUSA兼容，OpenFold MUSA扩展需要实际适配。没有静默CPU回退。

本仓库只上传参考评测，不包含模型优化实验或40%收益结论。原始源码快照在Release中保留；用于迁移的路径参数改动单独记录，不把新代码冒充原始测量代码。

来源与第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。实验功能评分只覆盖三个选定assay，不是ProteinGym全榜或已验证训练隔离测试。
