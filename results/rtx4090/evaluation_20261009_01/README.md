# RTX4090 · 2026-10-09 参考评测

先下载或克隆仓库，在浏览器打开 [report.html](report.html)。GitHub文件页展示HTML源码，不直接执行HTML。

- [metrics.csv](metrics.csv)：两模型正式性能行；保存median/P90/CV、吞吐、显存及设置。
- [experiment_index.json](experiment_index.json)：53项测量/诊断/捕获的完整settings；不把它们当53个同口径性能实验。
- [RESULTS.md](RESULTS.md)：原始目录职责和短名解释。
- [RAW_ARTIFACTS.html](RAW_ARTIFACTS.html)：13份Nsight原生报告、完整张量与trace的下载及校验。
- [publication_manifest.json](publication_manifest.json)：Git发布视图的来源与文件校验。

限制：BF16变长batch四项完整数值对照超阈值；Compute计数器受权限阻挡。OpenFold为B1/L191/预计算MSA/无模板/未松弛单体，不能外推为所有分支或其他GPU验收。

原JSON/日志中的服务器绝对路径和`verification.json`属于历史实验记录；发布链接另行检查，不修改这些旧运行事实。
