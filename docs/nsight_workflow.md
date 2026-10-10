# Nsight 原生评测

评测对象仅原版模型。先完成正式参考运行，再独立采集；正式benchmark不含profiler。

## 两个入口

```bash
# ESM3：四任务、batch以及warm/pipeline由配置定义
python scripts/run_nsight.py --suite <ESM3参考目录> --config <采集配置.json> --output <新目录>
# OpenFold：同卡通过的完整参考，独立环境
python scripts/run_openfold_nsight.py --reference <OpenFold参考目录> --scope warm --output <新目录>
python scripts/run_openfold_nsight.py --reference <OpenFold参考目录> --scope pipeline --output <另一个新目录>
```

Systems 使用 `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop`。容器限制下关闭CPU采样和上下文切换，不声称具有CPU堆栈归因。保存原生 `.nsys-rep`、SQLite导出、完整命令/日志、实际输出回放及源码hash。

warm只捕获准备并预热后的模型调用。pipeline捕获新进程首次模型加载、输入准备、预测、CPU/PDB交付；不称为冷磁盘启动，不包括未执行的外部数据库搜索等步骤。

`nsight_analysis.py` 对同名kernel按实际grid/block/registers/shared-memory拆分变体。计数器允许时，在实际NVTX operation中用完整kernel名称与name-filtered launch ordinal选定launch，再核对NCU原生报告的grid/block/registers；不把一个名字的所有shape混成同一工作量。

Compute sections为SpeedOfLight、Occupancy、LaunchStats、MemoryWorkloadAnalysis、ComputeWorkloadAnalysis；不锁频、不强制清缓存，实际设置保存到命令。`common/profiler_permissions.py`仅只读检查宿主策略与进程能力，不修改权限。受限或无法核验时不采集，指标保留null；如权限开放，使用新目录。

## 解释口径

- kernel累计占比：此名字的时长之和/全部kernel时长之和，不是wall时间占比。
- GPU activity：kernel/memcpy/memset在捕获区间内的并集占比，不是occupancy或NVML GPU utilization。
- 无GPU事件空档：不能自动归因Python、CPU、磁盘或调度。
- API时间可能与GPU重叠；cudaLaunchKernel可能包含反压，不全部称固定启动开销。
- memcpy的方向/字节来自真实事件；ATen copy_也可能是GPU精度转换，不能等同CPU→GPU。
- 未采到DRAM/cache/occupancy不能填0或画伪实测Roofline，也不能由kernel名字判断带宽/算力瓶颈。

当前权限仍为RmProfilingAdminOnly=1、容器无CAP_SYS_ADMIN。历史ERR_NVGPUCTRPERM日志作为拒绝证据保留；root身份不意味着有宿主计数器权限。

官方说明：[Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)、[计数器权限](https://developer.nvidia.com/ERR_NVGPUCTRPERM)。MUSA需目标SDK独立工具适配，CUDA采集不能改名作为MUSA证据。
