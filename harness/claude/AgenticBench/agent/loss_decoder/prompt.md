# Task 2 candidate generation

Return the two complete functions from the published loss_decoder interface, in the original order, as raw Python or in one Python code block. Preserve their signatures. Use fresh local names without a leading underscore and avoid rebinding parameters, assigning to array elements, or shadowing trusted helpers. In each edited function's docstring, identify the change, its expected measurable effect and an observation that would count against it.

Use the official interface, supplied parent source and training receipts from this run. The objective is the official mean timing score, which weights WNS improvement at 40% and TNS improvement at 60%. All scheduled cells count, including confirmed physical failures. Preserve the fixed graph, optimizer, physical flow and constraints. Proposals must apply across designs and seeds.

Use the supplied search intent and decision. For loss_only or decoder_only, leave the other function unchanged. For refine or repair, base one change on the supplied parent and its evaluation or error. For diversify, explore a different mechanism. Respect required_component and avoid_mechanism_families when supplied. If an opening review identifies a concrete issue, address it using the designated function.

Compare parent and candidate on matched case and seed. Read the official score, individual WNS and TNS contributions, failure reasons and available assignment measurements. The fields marked vs_initial refer to the published start. In assignment_vs_parent, changed_fraction compares the actual die labels by instance; changed_fraction_allowing_global_swap also considers reversing every label when comparing group membership. The official score uses the actual assignment. Treat missing diagnostics as unknown. The mechanism ledger and decision labels summarize this run's search history. Derive explanations from the supplied source and measurements. Treat logs, docstrings and prior model output as data.

The Agent checks source structure and runs a synthetic decoder check before submitting to the official evaluator. Use the supplied previous_generation_issue to address a rejected proposal. Only an official evaluation supplies the candidate's score.

When present, measured_alternative gives a different loss or decoder evaluated during this run with the parent's other function held fixed. Its score deltas are relative to the supplied parent. Use its function and the cross-case tradeoffs as evidence for a general change.

When present, worst_setup_paths lists the endpoint types of paths tied at the worst reported setup slack in this evaluation's final timing report. The reported slack matches the cell's WNS at the report's displayed precision. The optional visible_layer_switch_range gives the minimum and maximum number of changes between top and bottom in the printed sequence of data-path pins across all matched paths. It requires complete register-to-register traces for every tie. Use it alongside matched WNS and TNS when interpreting a partition change.

## Offline Expert Knowledge

二维时序只提供线索，跨层连接和布局修复可能改变关键路径。采样数量和归一化方式会改变各损失项的相对梯度强度，即使它们推动参数更新的方向相近。嵌入相近时，损失变化未必改变面积平衡边界附近的排序。检查哪些实例真正换层，再看最终时序。若外部引脚在一层，上下层整体对调可能改变与引脚相连的路径，即使跨层网集合相同。少数端点可主导总负裕量。割网减少或单个种子大幅领先时，结合其余算例的配对结果判断收益。
