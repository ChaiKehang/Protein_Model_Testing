# ProteinGym 标准数据：性能面板与功能质量分开

来源为 [ProteinGym 官方仓库](https://github.com/OATML-Markslab/ProteinGym)，参考表固定 commit `144fe22b07dfaeec2b366f2346203a9838a55b4c`，官方数据版本 v1.3。使用官方 ZIP，验证选中成员的 CRC、行数和本地 SHA-256；保留原实验标题/作者/年份等元数据。本地哈希不是发布者签名。下载与准备入口 `scripts/prepare_proteingym.py`；已固定数据在 `inputs/proteingym_v1.3_20261009/`。

## 性能与正确性

从参考表 217 个实验、186 个 protein ID 中，以元数据选择 10 个不同目标：五个预先固定长度桶每桶两个，排序 ID 后轮换 taxon 偏好。选择不使用模型预测或实验 fitness；全部目标序列为 assay 所提供的完整 context，有些是结构域/工程蛋白，不能一概叫完整天然 UniProt 蛋白。长度范围 47–1863。非标准字母和长度 >2048 的排除记录在 manifest。

使用原版模型、BF16、B1、seed7、3次预热/20次正式调用；四个不同长度目标另做真B4，并比较有效logits和embeddings。FP32额外诊断padding隔离和批次误差。不实施模型优化。

性能面板只检验这十个目标上的工程行为，不声称代表完整 217 实验，也不把 forward 正确性等同功能预测质量。

## 实验功能质量

选三个完整 assay，既包含单突变，也包含多突变，选择标准是任务类型/可解释性，不是模型得分：

- 酵母泛素 RL40A/Mavor：1253 个单突变，128 残基 target context，75 个被测位置。
- β-lactamase BLAT/Firnberg：4783 个单突变，286 残基 context，263 个被测位置。
- 蛋白 G SPG1/Wu：149360 个单/多突变，448 残基 context，四个被测位置。

采用官方 ESM baseline 文件 `proteingym/baselines/esm/compute_fitness.py` 的 masked-marginal 约定，适配本项目 ESM3 SDK。先把 WT 的一个被测位置 mask，得到该位置对所有序列 token 的 logits；评分为该上下文中「突变氨基酸 log probability − WT log probability」。同位置 log-softmax 分母抵消，因此直接 logits 相减等价。多突变对各位置分数求和，**不会直接刻画多个突变之间的联合相互作用/epistasis**。

所有 assay 行的 mutation 位置/WT 字母都应匹配 pinned target_seq，应用 mutation 得到的序列应与官方 mutated_sequence 完全一致；不静默丢弃失败行。一次完整 masked-context sweep 后，按初/中/末三个预定位置做 B1 与请求批次的全64logits数值诊断（FP32 atol=.001, rtol=.00001）。再独立测三次 warmed sweep，包括挑出该位置 logits 与复制小表到 CPU；排除 CSV 解析、变体查表、排名计算和落盘。

输出保存所有变体评分、masked-context 全64logits表、token IDs、输入 hash、完整配置和原始时间。吞吐单位是「masked contexts/s」：例如 GB1 的149360变体复用四次上下文预测，不能把149360/模型耗时冒充完整变体推理吞吐。

指标：Spearman 衡量实验 fitness 与模型分数的排序同向程度；AUC 使用官方二值标签，0.5 表示此排序无法区分两类，1表示完全区分。提供逐 assay 结果，**不计算或声称 ProteinGym 官方按蛋白/功能分类加权的完整总榜分数**。未设置事后质量通过线：PASS表示数据/执行/评分契约通过，相关系数是实际质量结果。

未核验 ESM3 预训练序列或实验信息重叠；这里 zero-shot 只表示没有用这些标签拟合或调参，不表示已证明独立泛化。OpenFold 的标准结构数据、S5000/A100 实测、GPU计数器和多卡验收仍需独立完成。
