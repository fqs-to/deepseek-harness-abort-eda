# 复现手册

## 11 月 19 日 · 十轮搜索的开跑前复盘

这次复盘使用团队归档中有完整九组正式训练回执的 28 轮。其中 10 轮相对父候选修改了解码器。租用机另一条主线已跑完 10 轮，自动选中第 10 轮，训练均分 32.6794。随后启动的独立版使用 11 月 19 日早上部署的代码。以下修订供后续独立运行使用，物理提分尚待测量。[队友的原始运行索引](../results/experiments-2026-11-18/README.md)

### 两个方向相反的误判

| 已有证据 | 合成输入的判断 | 正式训练观察 | 对 Harness 的影响 |
|---|---|---|---|
| branching 第 2 → 3 轮 | 八组小输入的每个标签都相同，旧检查会拒收 | 真实九组有三组改变，均分从 23.0823 升至 25.3459 | 合成输入用于检查形状、整数标签和两侧非空。标签恰巧一致的合法程序可以继续竞争。 |
| Claude 原版第 4 → 6 轮 | 八组小输入中有七组标签不同 | 两份程序的损失相同，解码器源码不同。九组 WNS、TNS 和计分逐项相同 | 改了代码、在合成输入上产生不同划分，都不足以证明值得再花一次物理评估。 |

第二个例子还有一项便宜的交叉检查。复用之前在三个公开**训练**设计算例上保存的 CPU 表示，对两个解码器使用相同输入。五份种子数据先逐实例对上已存的第 4 轮原生解码输出。两种解码器在 121,846 个超节点、353,315 个实例上均产生相同的合法化前分层。用第 2 轮作对照，同一工具检出了其中一份输入的 25,847 个实例全部翻面。这个观察针对五份 CPU 输入，物理时序的依据仍是历史九组正式回执。正式搜索只使用本次运行允许读到的输入与回执。[branching 第 3 轮](../results/experiments-2026-11-18/10820-branching/round-03/round.json) · [Claude 第 6 轮](../results/experiments-2026-11-18/15332-claude-original/round-06/round.json) · [重放保存的解码输入](../scripts/compare_saved_decoder_labels.py)

### 十次正式提交应当花在哪里

- **合约检查。** 先用官方语法接口和小输入检查明显无效的程序。已评程序继续按当前账本与去文档字符串后的语法树去重。历史回放中修订版放回了一份曾被误拒的有效提案；这些数据不能估计新搜索最终会提高多少分。
- **开局评审。** 模型明确要求所有备选重写时，这批有缺陷的方案直接退出；只有评审已选出可用方案时，另一份合格但未测的方案才进入候补。原版 34.75 那次的两次评审都选择了 A，该风险通过构造评审与重启日志复现。
- **保持搜索与计量一致。** 本次生成的源码与原请求 ID 一起保存。遇到暂时的通信中断先查账本，受理过的请求继续等原结果。模型连续生成无效程序时，先考虑已看过但未评的有效程序，再按剩余模型预算放宽搜索方向。源码接口、去重和费用额度在恢复中依然生效。
- **训练内反馈。** 先前版本已能比较同一设计、同一种子的上下层分配与时序，在本次运行中组合分别有效的损失函数与解码器，并用一个公开训练批次比较生成的两个开局损失对梯度的影响。选定候选始终依据本次完整正式训练回执。公开验证只用来报告选定后的泛化表现。

团队按十轮安排正式搜索；比赛给每次正式运行设定 10 次候选提交和 15 元模型费用上限。联调和复盘在正式账本之外完成。运行从官方程序开始，离线领域经验集中写在初始提示词的专家知识段，按题面规则连同标题不超过 200 个计数单位。团队旧候选、针对具体设计或种子的调参记录不进入新 Agent 的提示或安装目录。

在仓库根目录执行以下命令可复查解码器历史回放和搜索回归。公开归档、复盘工具和测试均在参赛 Agent 目录之外。

本次在官方 Docker 镜像、Python 3.8 和一个真实公开训练批次上跑过全部 60 项测试，耗时 21.4 秒。十轮流程使用模拟账本验证。在连续 12 次收到无效提案的用例中，第 13 次生成恢复，最终有 10 个不同的评估请求被受理。正式物理搜索的结果需等该版本独立跑出回执。

```bash
python3 -B harness/claude/tests/replay_archived_decoder_probes.py /path/to/contest-source/AgenticBench/local_tasks/tp_gnn_loss_decoder_timing/initial_program.py
python3 -B -m unittest discover -s harness/claude/tests
```

## 先跑离线回放

用 Python 3.10+，在仓库根目录执行：

```bash
python3 -m pip install numpy
python3 -B -m unittest discover -s harness/claude/tests -v
```

回放覆盖候选搜索、失败分类与选择。没有 NumPy 时解码探针测试会跳过。这一步不需要 Docker、GPU 或 DeepSeek API，也不代表拿到了相同的物理分数。

## 再在独立目录做真实评估

1. 在 Linux + NVIDIA GPU 主机准备 Docker、官方任务包和 Python `PyYAML`。按来自官方的 `contest-source/README.md` 解压源码与数据、加载镜像。
2. 在一份**新解压**的 `contest-source` 安装本仓库的 Harness。安装脚本从该任务包恢复 `program.py`，不要拿历史候选覆盖它。
3. 检查环境，然后用空的 `--work-root` 开新实验。同一实验恢复时使用原目录和原配置。API key 由运行者经 `DEEPSEEK_API_KEY_FILE` 设置，密钥文件放在代码、数据和工作目录之外。

```bash
python3 scripts/install_harness.py /path/to/contest-source
cd /path/to/contest-source/AgenticBench
python3 -m contest2 check --baseline loss_decoder
python3 -m contest2 run --baseline loss_decoder \
  --work-root /path/to/contest-work/claude-handoff-01 \
  --retain-physical-files
```

`check` 无误后再开始收费的演化。训练每个候选一轮，完整九组；单次运行按当前任务包由外层计量候选次数和 API 费用。两套 Harness 做对照时分开工作目录，保持官方包、GPU 类型和固定种子一致。若要保存完整布局文件，先给工作盘留足空间再打开保留开关。

## 归档赢家，再补验证

从这次运行的 `session-result.json` 读取最终选中的候选 ID。保留同一工作目录里的 `candidates/<id>/candidate.json`、训练矩阵回执、九组结果及日志；单独导出其中的源码。检查 ID 与训练矩阵匹配后，再对同一个候选跑六组公开验证：

```bash
python3 -m contest2 validate --baseline loss_decoder \
  --work-root /path/to/contest-work/claude-handoff-01 \
  --candidate-id "$CANDIDATE_ID" --retain-physical-files
```

验证需要原工作目录中的真实训练记录，不能把另一次运行的候选 ID 直接代进去。验证结果用于报告泛化表现，不反馈进这次正式搜索。整理交付时保存原始回执、布局、指标与时序报告，以及对应的 Harness 版本和候选源文件。
