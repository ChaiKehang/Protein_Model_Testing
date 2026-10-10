# 重建环境、复跑与跨卡比较

本说明针对 Linux GPU 服务器；当前公开结果实测 RTX4090，A100/S5000 未实测。上传前仅进行了代码语法、结果来源、链接及比较入口检查，没有在本机 Windows 重跑模型。

## 1. 克隆与目录约定

```bash
git clone https://github.com/ChaiKehang/Protein_Model_Testing.git
cd Protein_Model_Testing
mkdir -p references cache outputs
```

`results/` 是已发布精简基线；新测量写 `outputs/` 下新目录。权重、Python环境和原生工具不在Git中。原JSON保留当时服务器路径，这是溯源记录，不必创建相同服务器目录。

## 2. ESM3环境与固定权重

参考环境：Python3.12、esm3.2.3、Torch2.3.1+cu121、numpy1.26.4、transformers4.48.1、huggingface-hub0.28.1、biotite1.2.0。关键约束在 `configs/constraints-cuda.txt`。

```bash
conda create -n esm3-eval python=3.12 -y
conda activate esm3-eval
python -m pip install 'setuptools<81' wheel
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
git clone --branch v3.2.3 --depth 1 https://github.com/Biohub/esm.git references/esm
python -m pip install -c configs/constraints-cuda.txt -e references/esm
python -m pip install matplotlib scipy
python -m pip check
cp configs/esm3_weight_manifest.json model_snapshot.json
python scripts/download_weights.py
```

下载器沿用manifest中的固定revision，并更新实际snapshot路径、核验必需文件和记录SHA-256。若官方模型要求登录/访问许可，按其条件完成；不要提交token。首次完整运行前可用 `main.py --verify-weights` 重新核验权重。

```bash
python scripts/restore_proteingym_inputs.py
python -m unittest discover -s tests -v
python main.py --task forward --sample-id random_L256 --batch-size 1 \
  --manifest model_snapshot.json --device cuda:0 --precision bf16 \
  --runs 20 --warmup 3 --end-to-end-runs 2 --output outputs/forward_smoke
```

三个DMS输入CSV从官方archive恢复，检查冻结archive和逐assay SHA-256；已有数据不一致时拒绝覆盖。`inputs/structures/` 已保存1UBQ与1CRN小型结构输入。

## 3. OpenFold独立环境

参考为Python3.10/Torch2.5.1+cu124，官方commit `be2ec1841f16c966c65ae0e7599ebbadc725757d`。建立独立环境，不与ESM环境混装。

```bash
conda create -n openfold-eval python=3.10 -y
conda activate openfold-eval
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/aqlaboratory/openfold.git references/openfold
git -C references/openfold checkout be2ec1841f16c966c65ae0e7599ebbadc725757d
# 根据固定源码的README/environment.yml安装其依赖并构建CUDA扩展。
# 扩展必须针对当前目标GPU与实际CUDA编译环境构建，不复制4090的SM89二进制。
python -m pip install --no-build-isolation -e references/openfold
```

该固定源码的扩展安装需要可用的CUDA编译工具、兼容编译器和Python构建依赖；上面不是免前置依赖的一键安装。4090实测使用nvcc12.8/GCC11.4；驱动、Torch内置CUDA与扩展编译工具分别记录，换卡先检查它们兼容。安装失败不删网络层数或减少MSA/recycling来替代。

从 `inputs/openfold/weight_manifest.json` 的官方S3来源下载 `finetuning_no_templ_ptm_1.pt`，SHA-256必须为：

```text
1a179937a6f61143c41f8b6a0b763ac1f4762c9dc63d2fb151240930c110bee1
```

输入使用该固定OpenFold源码的 `examples/monomer/fasta_dir/6kwc.fasta` 与 `examples/monomer/alignments/6KWC_1`。详细预测、CPU数值参考与工具前提见 [OpenFold契约](gpu_optimization/openfold_reference.md)。

## 4. 完整串行评测

准备完上述两个环境、权重和数据后，显式填写两个Python绝对路径及实际OpenFold权重路径：

```bash
ESM_PY=/你的环境/esm3/bin/python
OF_PY=/你的环境/openfold/bin/python
RUN=outputs/a100_reference_日期_01
"$ESM_PY" scripts/run_evaluation.py --output "$RUN" --device cuda:0 \
  --esm-python "$ESM_PY" --openfold-python "$OF_PY" \
  --esm-manifest "$PWD/model_snapshot.json" \
  --dataset "$PWD/inputs/proteingym_v1.3_20261009" \
  --esm-pdb "$PWD/inputs/structures/1UBQ.pdb" \
  --crambin-pdb "$PWD/inputs/structures/1CRN.pdb" \
  --openfold-weights /实际路径/finetuning_no_templ_ptm_1.pt \
  --openfold-fasta "$PWD/references/openfold/examples/monomer/fasta_dir/6kwc.fasta" \
  --openfold-alignments "$PWD/references/openfold/examples/monomer/alignments/6KWC_1"
"$ESM_PY" -m unittest discover -s tests -v > "$RUN/unit_tests.log" 2>&1
"$ESM_PY" scripts/build_evaluation_report.py --output "$RUN"
"$ESM_PY" scripts/finalize_evaluation.py --output "$RUN" --tests "$RUN/unit_tests.log"
```

未安装Nsight或无权限时查看保存的错误/缺失状态，不把CUDA采集替换为MUSA。完整CUDA套件依赖实际可用的`nsys`/`ncu`；Compute计数器受限时流程保留null。S5000尚需OpenFold扩展和MUSA profiler实际适配，当前入口会明确拒绝未适配路径。

上传仓库相对原测量源码的变化仅涉及可配置输入路径及新增下载/比较入口；实际推理适配器与Backend未改变。基线Release保存测量时源码，新的目标卡每次运行保存自身源码hash。

## 5. 跨卡比较

先检查完整数值/结构与任务质量，再解释性能变化。精简Git结果可比较设置与指标；完整张量对照需要从Release恢复对应原始证据。

```bash
python scripts/compare_run_metrics.py \
  --reference results/rtx4090/evaluation_20261009_01 \
  --candidate outputs/a100_reference_日期_01 \
  --output outputs/comparison_4090_a100.csv
```

比较器检查任务、B/L、精度/计算策略、seed/生成设置、权重、输入、相关源码、计时范围、预热和正式次数；不一致时标记 `not_comparable`，不输出伪加速比。`latency_speedup = 4090 median / 目标卡 median`；大于1表示目标卡更快。配置匹配不代表输出正确。

完整ESM输出用 `compare_outputs.py --help`；OpenFold用 `scripts/compare_openfold.py --help`，并沿用事先声明阈值。不要只比较随机生成字符串，也不要把单算子加速推广成整个pipeline加速。
