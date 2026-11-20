# 赛题 2 · 3D 芯片网表切分交接

[11 月 20 日最终版及选版对照](results/final-2026-11-20/README.md)汇总十轮正式训练、六组公开验证、所选算法和逐组计分，可与 Claude 原版及队友的固定算法实验对应核对。

我们使用官方 `loss_decoder`（TP-GNN 表征学习与聚类）。截至 2026-11-18 的已归档结果，Claude 原版第 4 轮九组训练得 **34.75 分**，该候选的六组公开验证得 **−4.35 分**。11 月 18 日的分支搜索版已取得 **26.93 分**，Claude 第三版已取得 **29.67 分**，这两个数字均为完整九组训练分。

[逐次 Harness 实验归档](results/experiments-2026-11-18/README.md) 包含各版本源码、候选程序、模型可见交互、提交决策，以及按算例和种子拆分的训练与公开验证结果。各运行页注明快照时间和缺失记录。

| 队友要做什么 | 从这里开始 |
|---|---|
| 了解赛题和两条官方基线、下载官方包 | [官方基线说明](docs/official-baselines.md) |
| 安装 Harness、启动检查、复跑和验证 | [复现手册](docs/runbook.md) |
| 查看最优候选、分数和归档状态 | [第 4 轮源码](candidates/claude-round-04/program.py) · [实验记录](results/2026-11-17.md) |
| 对齐分工、验收和交付先后 | [团队计划](docs/team-plan.md) |
| 看实跑 Harness 的实现 | [Harness 源码](harness/claude/AgenticBench/agent/loss_decoder/) |

**两种文件用途不同：** Harness 的运行起点是官方 `initial_program.py`，安装时从官方任务包恢复；`candidates/` 保存跑出的算法，供对照、复现和提交整理。候选文件不复制进正式 Harness 的起始目录，也不作为模型输入预置。

官方源码、数据包和镜像请按[赛题页面](https://www.aicomp.cn/tracks/tracks-4/5590.html)和[参赛流程](https://www.aicomp.cn/process)获取。提交做法以当前官方任务包和赛事通知为准。
