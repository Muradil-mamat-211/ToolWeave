# RODS 验证策略：2026-10-08 调整

默认采用 `validation_policy: rods` 和 `replay_final_gt: false`：在整段润色和 MF/MP 变换之后，用构建时的真实执行证据做一次完整 Quality Judge 审查。此前的严格验证器保留为可选 `strict` 模式。

`rods` 是沿用的配置名。取消默认最终 GT 重放是用户要求后的项目简化，与论文附录 G 的三个确定性检查流程不同；不声称默认完整复现论文。设 `replay_final_gt: true` 可保留附录 G 的额外新 VM 重放，strict 始终重放。

## 查证依据

RODS 论文附录 G 描述三个确定性准入检查：新 VM 重执行、工具可用性、参数复杂度（字符串至多 200 字符、列表/元组至多 5 项）。随后 Quality Judge 检查 Query 与 GT、状态、跨轮关系、自然度和类别结构；拒绝后最多一次 Query 修复和第二次 Judge。逐轮 Query Verification 另见附录 D。[RODS 原论文](https://arxiv.org/html/2606.19047v1#A7)

公开 [RODS 目录](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS) 在查证时只有 README 和 assets，没有可核对的生成器源码。查证的上游 HEAD 为 `be52dbf33051c9b86e8e4d3c4e2394548906c75b`；目录清单保存在本次审计目录的 `rods_listing.json`。这里对齐的是论文公开流程，项目实现与补充提示词仍标记为项目设计。

## 当前实现

| 检查/组件 | 默认 rods | 可选 strict |
| --- | --- | --- |
| 最终新 VM 重执行 | 默认跳过；`replay_final_gt: true` 时启用 | 必须通过 |
| 工具可用性、参数复杂度 | 程序检查，必须通过 | 必须通过 |
| 每轮 Query Verifier | 保留 | 保留 |
| 最终 Quality Judge 与最多一次 Query 修复 | 保留 | 保留 |
| 独立 Final Query Verifier | 最终语义交给 Judge，不重复调用 | 保留 |
| 类别、单位、参数来源、缺失参数有效性、Observation 支持、关系定位、文本契约、动作必要性 | 非阻断诊断 | 任一失败则拒绝 |
| 固定文本模板语法、LC 长度/跨度/末尾位置阈值 | 不作为统一提示词要求或准入门槛 | 保留原策略 |
| 输入参数 schema、实际执行错误、生成泄漏、候选 schema 与 Training 契约 | 保留 | 保留 |
| 原始模型响应、GT、VM 返回 | 不修改 | 不修改 |

诊断在对话草稿的深拷贝上计算，防止参数依据提取把实际草稿写成 `UNSUPPORTED` 后影响 Judge。原始 `passed=false` 结果保存在 `generation_metadata.project_diagnostics`，并标记 `blocking=false`；诊断异常记为 `passed=null`。必需 gate 列表只包含当前政策的真正准入检查，绝不将失败结果伪装为通过。Refine 后另记新阶段的诊断。

默认不产生 `fresh_vm_gate`，也不记录一个伪造的重放通过结果。候选元数据明确记录 `final_gt_replay_performed=false` 和 `execution_validation_source=synthesis_vm_trace`；真实构建记录仍保存在 `execution_trace`。显式重放或 strict 时记录实际的 fresh gate，并标记 `fresh_vm_replay`。参数大小和工具恢复检查不产生模型调用。

最终审查中的 GT 是**新数据先构建、执行成功的调用链**。重新执行这些调用只能证明可运行，不能证明后写的 Query 合理。Judge 的输入现在包含所有 Actor 可见工具的定义（使用与未使用的工具分别列出）及逐轮可用工具名称，从而能够检查 MF 是否可被其他工具完成。没有引入全局 128 工具列表。

MF 审查要求受影响请求确实无法通过当轮工具完成，恢复工具后原调用链完成此前未解决的请求；只有工具更新而没有新 Query 的恢复轮是合法的。MP 审查要求必要值在当前请求与此前可见历史中确实缺失或歧义，澄清轮提供该值并支持恢复调用。未来澄清、隐藏状态和 Planner 叙事不构成当轮可见信息。缺失轮应为 `GT=[]` 且没有执行 Observation；这不是 Query-GT 不一致的自动证据。

MP Transformer 仍检查所选参数是否必需、缺失轮是否暴露值、澄清轮是否补值，并构造合法空 GT/恢复 GT。其额外的“已有上下文是否唯一可恢复”启发式在默认模式只记录诊断，类别语义交给最终 Judge。

Planner 的输入 JSON、类别区分、函数范围、输出标签与轮数预算保留。默认提示词允许明确的历史取值、普通计算与简洁组合，要求实际目标与 GT 一致，不再为所有文本任务强制同一种 exact-format 语法。Judge 仍能拒绝错误数值、没有依据的请求、机械任务与不真实的澄清。

配置 `stage1_format_rl/configs/rods_data_generation_codex_cli.yaml` 显式设为 `rods` 和 `replay_final_gt: false`；模型后端仍为官方 `codex exec` 子进程，不引入项目 LLM API/SDK。输入导出脚本支持 `--validation-policy rods|strict`，默认导出 rods 提示词。

## 已有三个真实例子的重放

最初离线验证阶段没有恢复生成，也没有调用新 Judge。以下结果来自原始 `draft_rewritten_1.json` 的新 BFCL VM 重放；随后用户授权的真实 Judge 结果见下一节。

| 原数据 | 新 VM 调用数 | 新 VM / 工具可用性 / 参数复杂度 | 最终 Judge |
| --- | --- | --- | --- |
| `multi_turn_base_155` | 5 | 三项通过 | 暂停，未运行 |
| `multi_turn_long_context_73` | 3 | 三项通过 | 暂停，未运行 |
| `multi_turn_long_context_158` | 4 | 三项通过 | 暂停，未运行 |

三个例子的旧 `semantic_grounding_gate` 仍显示失败；两个 LC 还触发旧文本契约等诊断。新策略允许它们进入语义审查。这不是三个最终候选已接受的证据，也不是四类或 12 条的整体成功率。特别是 LC 73 的摘录收据是否自然、是否有实际用途，仍应由 Judge 审查。

完整证据与无模型重放脚本：

- `/mnt/data/toolweave_grounding_relaxation_20261008/rods_core_replay.json`
- 同目录的三份 `*_fresh_vm_trace.json` 与 `replay_rods_core.py`
- 同目录的 `planner_inputs_rods/`：五个原 seed、覆盖四类的实际输入与提示词，模型调用数为 0。

历史 12 条测试的 verdict、候选队列与暂停标记保留原样。重新使用新政策做真实模型测试时，应另外记录政策、提示词与代码版本，不能把这次离线重放并入旧测试作为通过。

## 用户随后授权的三条真实 Judge 审查

2026-10-08 16:45–16:49（北京时间），仅对上述三条原始 draft 调用现有 `codex exec` 后端的 Quality Judge，没有恢复批量生成，也没有 Query 修正或重复审查。结果如下：

| 原数据 | Judge | 当前政策最终验收 |
| --- | --- | --- |
| `multi_turn_base_155` | accept | 三项基础检查、Judge、候选 schema 与 Training 契约通过 |
| `multi_turn_long_context_73` | accept | 三项基础检查、Judge、候选 schema 与 Training 契约通过 |
| `multi_turn_long_context_158` | reject | 第 2、3 轮出现字典操作和接口字段提取的技术措辞，被 Judge 判为不自然及接口泄露 |

LC 158 的数值、真实执行和历史关系被 Judge 认可；拒绝原因是 Query 表达。两个通过候选仅保存供审阅，没有写入 Training 队列，也没有运行新的 Actor rollout。完整输入、原始响应、三个判决与候选契约复查位于 `/mnt/data/toolweave_existing_three_judge_20261008/run_20261008T084540_801661Z/`；中文报告为同目录 `Judge结果.md`。

## 此前验证范围

新增政策测试覆盖四类真实 CPU VM 流程、三项必需检查的阻断、失败诊断的保存与隔离、Judge 解析失败、一次修复上限、GT 无法修复的丢弃和额度延期；模型回复由 FakeLLMBackend 提供，不计为真实 Codex 质量证据。原严格策略测试仍显式采用 strict。

本次相关回归共 235 项通过，其中新增政策测试 16 项。另有一项历史恢复测试所需的 `stage3_generator_incident_recovery_20260812T071318Z/queues/validated_candidates.jsonl` 在当前工作区缺失，未完成该项验证；没有伪造替代文件。JUnit 位于本次审计目录的 `pytest_results.xml`。

## 取消默认重放后的回归记录

本次修改后的政策与 pipeline 测试 39 项全部通过，覆盖四类真实 CPU VM 构建、默认不调用最终重放、MF/MP 空 GT 与恢复调用不变、一次正常 Judge、真实工具恢复失败的阻断、显式重放开关、拒绝修复上限及严格模式兼容性。模型响应使用 FakeLLMBackend，不代表新 Judge 提示词的实际判别准确率。

扩展兼容性检查为 198 项通过、1 项失败；合计 237 项通过。失败为 `test_stage3_training_data_has_four_original_types_and_current_protocol`：本地训练资产 400 条原始提示词均没有该测试要求的 `<think>/<answer>` 协议头。按样本 ID 比较，这些提示词与原始 `bfcl_train.parquet` 完全一致；其原始 SHA256 为 `f6c9311b3b7977b307fcb8d175603e828dbca0542bb3abca4f4bdf922bc8f293`。本次没有修改训练资产。前述缺少历史文件的恢复测试仍单独排除，未计为通过。

本次没有新增真实模型调用，没有恢复 12 条批量生成，没有写入 Training 队列；此前三条真实 Judge verdict 保留，不能当作新提示词的测试结果。新回归 JUnit、资产问题核对结果与代码指纹位于 `/mnt/data/toolweave_final_query_review_20261008/`，汇总为 `manifest.json`。
