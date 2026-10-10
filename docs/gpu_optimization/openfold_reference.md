# OpenFold 4090 参考契约

本次补充任务书 P0 中的 OpenFold 结构预测入口。固定官方代码 commit `be2ec1841f16c966c65ae0e7599ebbadc725757d`（setup 标记版本 2.2.0）。独立环境 `/root/autodl-tmp/envs/openfold-cuda`，Python 3.10、PyTorch 2.5.1 CUDA 12.4；不升级 ESM3 的 Python 3.12 / Torch 2.3.1 环境。

- 模型：官方 `model_3_ptm` 模板关闭预设；官方 `finetuning_no_templ_ptm_1.pt` 权重。
- 权重来自官方脚本指定的公开 S3 bucket。只下载这一份；来源、字节数、服务器 ETag 和本地 SHA-256 在 `inputs/openfold/weight_manifest.json`。本地哈希不是发布者签名。
- 输入：固定上游仓库的 6KWC FASTA 与 `examples/monomer/alignments/6KWC_1` 预计算 MSA。完整 MSA 解析、特征转换、48 个 Evoformer blocks、默认三个 recycling steps（四遍主干）及结构头；没有通过减层数、MSA 上限或 recycling 数来制造运行成功。
- FP32，CUDA matmul/cuDNN TF32 关闭。所有设备使用固定 chunk=64、关闭自动 chunk 调优，便于相同任务比较。这是显式参考配置，不是声称完全照搬默认 chunk=4 的性能；各卡参考评测应保存同一配置。
- 正式计时：模型及设备特征已准备；同步完整模型调用；加载/准备/第一次推理单独记录，PDB 与 .pt 输出不计入 warmed latency。profiler 额外运行，时间不能替代 benchmark。
- 范围：模板关闭的单体模型、预计算 MSA、未松弛 PDB。没有证明 MSA 数据库搜索、模板路径、OpenMM relaxation、多体模型或标准数据集整体质量。
- 结构质量：RCSB 6KWC mmCIF；先全局序列对齐，再用同字母、双方有效的 Cα。记录覆盖率与索引，不把缺失残基当零坐标。计算 Kabsch RMSD、15 Å 邻域的 Cα lDDT，并与固定版本 OpenFold 的 `lddt_ca(..., per_residue=False)` 对照；pLDDT 只叫模型置信度。
- CPU-FP32 / CUDA-FP32 的探索阈值已在 `configs/openfold_reference.json` 冻结：输入特征哈希与 atom mask 相同，Cα 对齐 RMSD ≤0.05 Å、pLDDT 最大差≤0.5。未测之前不能填通过；它们不是最终 MUSA 验收标准，也不是实验结构精度门槛。
- MUSA 明确拒绝未经移植的 OpenFold CUDA extension 路径；不隐式 CPU 回退。须在 S5000 环境检查实际算子与扩展、建立真正移植后才运行。

执行示例（参考代码已安装、权重已校验后）：

```bash
/root/autodl-tmp/envs/openfold-cuda/bin/python scripts/openfold_baseline.py \
  --weights /root/autodl-tmp/openfold-cache/params/finetuning_no_templ_ptm_1.pt \
  --fasta /root/autodl-tmp/references/openfold/examples/monomer/fasta_dir/6kwc.fasta \
  --alignments /root/autodl-tmp/references/openfold/examples/monomer/alignments/6KWC_1 \
  --device cuda:0 --chunk-size 64 --warmup 1 --runs 5 --profile \
  --output outputs/openfold_cuda_<日期>
```

CPU 对照应使用该输出的 `--features <CUDA目录>/features.pt` 和显式 `--attention-kernel torch`，同时校验配置与输入哈希，避免 MSA 重采样导致不等价输入。默认 ExtraMSA 路径实际调用 CUDA 扩展，最初 CPU 运行失败已保留在 `outputs/openfold_cpu_20261009_01`；不是 CPU 正常运行的证据。

`torch` 策略通过每个官方 Attention 实例的 pre-hook 将 `use_memory_efficient_kernel` 设为 False，选择固定版本源码已有的 matmul/softmax 分支；保留 q/kv/mask/bias、网络、权重和特征，不把模型移到其它设备。位置参数和关键词调用、真实非零输出以及隐藏位置不影响结果已在独立环境 `tests/openfold_checks.py` 检查。CPU 与 CUDA attention kernel 实现明确不同，需要完整输出误差检验，不能要求或声称 bitwise replay。

另一次完整 CPU 排错在结构模块 IPA 找到第二处 CUDA softmax 扩展调用，失败保留在 `outputs/openfold_cpu_torch_20261009_01`。torch 策略也将官方 IPA 的 `inplace_safe` 显式设为 False，选择其现有 torch softmax 与点聚合分支。这会改变点聚合的求和顺序，必须比较完整结构输出；两处纯 CPU 路径的小算子检查不等于模型通过。

CPU 数值参考用 `--numerical-only`：保留一次完整预测与所有输出，不重复跑 CPU 性能实验；报告 warmed latency 为 null。本阶段只记录原版性能，不做优化收益验收。

结果状态以各 `report.json` 为准，脚本、安装成功或权重文件存在不等于推理完成。

## 原生采集入口

先在实际目标 NVIDIA 卡上生成通过的参考报告，再用该硬件的解释器运行：

```bash
python scripts/run_openfold_nsight.py \
  --reference outputs/openfold_<目标卡参考目录> \
  --device cuda:0 --output outputs/openfold_nsight_<新目录>
```

该入口保存原生 `.nsys-rep`、显式 SQLite 导出、实际 module/kernel 形状与 representative launch 计划；使用原始 NVTX 范围、name-filtered launch ordinal 选择 Compute，并核对采到的 grid/block/registers。未开放计数器时保留 null 和实际策略，不尝试修改宿主配置。原生采集发生于暖模型预测；不包括 MSA 搜索、数据准备、加载或结果交付，也不能替代正式计时。MUSA 需要目标 SDK 的独立原生 profiler 适配。
