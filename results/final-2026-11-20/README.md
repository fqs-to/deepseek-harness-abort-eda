# 赛题二最终版与 Claude 原版

三组训练算例和两组公开验证算例各有三个种子。下表的五算例均分按算例等权，相当于训练与公开验证按 3∶2 加权。赛题正式成绩还包括两组隐藏算例，由组委会评测。

| Harness | 自动返回轮次 | 九组训练均分 | 六组公开验证均分 | 五算例均分 |
| --- | ---: | ---: | ---: | ---: |
| 最终版 | 6 | 35.3374 | -0.1265 | 21.1518 |
| Claude 原版 | 4 | 34.7550 | -4.3459 | 19.1146 |

最终版在五个已知算例上领先 2.0372 分。

最终版正式账本记录了 23 笔模型请求，模型费用为 1.39987856 元，限额为 15 元。费用不含 GPU 租用。

## 队友固定算法的离线评测

队友在本机用同一份算法评测了全部五个公开算例的十五组种子。这是固定算法的离线开发评测，尚无正式十轮会话返回该算法的回执。已核对算法源码和十五组的上下层最终 DEF、切分、指标及分层时序报告，十五组分数均按共同参考 WNS、TNS 重算。[测量脚本](https://github.com/SKTGAN/aic-eda-task2-experiments/blob/54eff11293a59c942631803ea9fc95e6c230acad/local-experiments/fifteen-20261120/measure_fifteen.py) · [算法源码、分数和物理文件](https://github.com/SKTGAN/aic-eda-task2-experiments/releases/tag/physical-layouts-20261120)

| 计分范围 | 最终版 | 队友固定算法 | 最终版减固定算法 |
| --- | ---: | ---: | ---: |
| 九组训练均分 | 35.3374 | 41.7328 | -6.3954 |
| bp_fe-ng45 | 46.8630 | 55.1408 | -8.2778 |
| bp-ng45 | -14.0874 | -2.4343 | -11.6531 |
| ariane133-ng45 | 73.2365 | 72.4918 | +0.7447 |
| 六组公开验证均分 | -0.1265 | -10.7083 | +10.5818 |
| swerv_wrapper-ng45 | -11.7243 | -4.6421 | -7.0822 |
| bp_fe-asap7 | 11.4712 | -16.7745 | +28.2457 |
| 五算例均分 | 21.1518 | 20.7563 | +0.3955 |

## 对照与归档

最终版的十轮与各组计分见 [最终版回执摘要](final-results.json)。单元状态区分合法结果与物理失败罚分，得分 −100 的合法结果仍保留成功状态。账本中的未成功请求保留其状态和空分数。

正式返回候选为 `7196c38fd3f632fb5a2abf144031b37f8a0af26b3d8377116317b024193b5f17`。[物理算法](algorithm/program.py) 与该候选源码一致。团队另行保存九组训练及六组公开验证的原生物理包，SHA-256 为 `41345a9bb0c4ca1dc8c56df460a4838daa2a4c9ed3199e604a1be59cef7d5c71`。

Claude 原版按[第四轮选择记录](https://github.com/SKTGAN/deepseek-harness-abort-eda/tree/1e7c625d68f4e0679bc325eb37b257423aa0a235/results/experiments-2026-11-18/15332-claude-original)与[公开回执摘要](claude-comparison.json)核对。队内收到的原生 session-result.json 确认同一候选。交接 ZIP 有十五组原运行指标，原次最终布局为 0/15，另附三组训练复测布局。学校机另存九组训练复现布局，均分 16.5377，其中三组与原分相同。

## 最终版 Harness 还原

在官方工具包与本结果目录所在的位置执行下列命令。官方起始算法从任务包提取，团队的 Harness 文件从本目录覆盖进去。独立运行时使用新的工作目录和赛方配置。

```sh
tar -xzf contest-source.tar.gz
printf '%s  %s\n' 'f18b6e303cad3e67b7dd5be6a63a2ca7399e43997fd0c900334adae1a31bc87b' 'contest-source/AgenticBench/agent/loss_decoder/program.py' | sha256sum -c -
cp harness/* contest-source/AgenticBench/agent/loss_decoder/
cd contest-source/AgenticBench
python3 -m contest2 check --baseline loss_decoder
python3 -m contest2 run --baseline loss_decoder --work-root "$WORK" --retain-physical-files
ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_id"])' "$WORK/session-result.json")
python3 -m contest2 validate --baseline loss_decoder --work-root "$WORK" --candidate-id "$ID" --retain-physical-files
```

执行前将 `WORK` 设为空的工作目录，并按本机的官方配置指定数据和运行环境。
