# Planner v2 验证范围

本文件记录实际检查，不把人工格式示例、脚本化模型测试或真实模型测试混为一类。设计和配置见 [PLANNER_V2_DESIGN.md](PLANNER_V2_DESIGN.md)。

下方四类 11 项验收是较早版本的历史。严格复审发现自由摘要与固定 GT 的标签歧义、粘贴历史答案以及 LC 只使用开头信息。新增 `query_contract_gate` 重新检查时，四条旧候选全部拒绝。后续严格策略与实测单独记录在 [QUERY_CONTRACT_HARDENING.md](QUERY_CONTRACT_HARDENING.md)，不可用旧通过状态代替新版质量证据。当前默认政策见 [RODS_VALIDATION_POLICY.md](RODS_VALIDATION_POLICY.md)，公开候选见 [数据包](../../data/codex-synthesis/README.md)。

新版最终四类各一例已通过全部 12 项验收、真实独立 VM、最终 verifier/Judge 和 Candidate/Training 契约，回归为 248 passed、1 deselected，17 个诊断负例均拒绝。实际新数据、一次 Base Query 修复、原始失败及限制见上述修订记录与 [公开四类数据与原始验收](../../data/codex-synthesis/README.md)。

## 自动化回归

覆盖四类 Generator 流程、真实 BFCL VM 执行与独立重放、原工具恢复事件、输入白名单和 schema、轮数预算、缺失轮执行记录归属、队列/检查点的根顺序、CLI 子进程传输与超时、语义检查、参数预算前置校验、候选与 Training 契约、队列恢复协议。

核心 Generator 流程测试的模型返回是脚本化 fixture；VM 使用真实 BFCL 实现。它们能检查代码分支、数据结构和验收行为，不能估计真实模型的生成成功率。

唯一未运行的历史测试为 `test_real_recovery_candidates_are_reclassified_by_evidence_not_seed_hardcode`，因为其所需的历史 `validated_candidates.jsonl` 不在本次环境中。没有用新造文件替代历史证据。

测试数据从实际训练 parquet 导出，保留任务、GT 和 schema；不是另行下载的独立 BFCL checkout。公开验证 parquet 原文件另存为 `bfcl_val_upstream.parquet`，含 100 条任务；测试兼容副本只替换系统 Actor 协议为本仓库的已有协议，任务内容与 GT 保留。兼容文件名含 `400` 不代表其实际有 400 行。

本次工作区的 JUnit 结果为 `/mnt/data/toolweave_planner_v2_audit/pytest_results.xml`，数据来源与哈希见该目录的 `assets/manifest.json`。最终统计在下方实测完成记录中汇总。

## 不调用模型的真实输入验证

`export_planner_inputs.py` 已读取整个 400 条训练任务，逐条构建实际输入、校验 schema、创建初始 BFCL VM 并导出快照视图和 prompt，模型调用数为 0。输出目录为 `/mnt/data/toolweave_planner_v2_audit/current_inputs/`。

已验证导出工具可以直接运行，不需测试用的 namespace bootstrap。Generator 的 lifecycle/provenance 导入不再强制加载 Training 的 torch/scipy；Training 原公开导出仍按需解析到相同实现，另已检查公开导出和 exact-match 行为。

配置 `rods_data_generation_codex_cli.yaml` 已通过实际 loader，选择 `codex_cli`、继承 CLI 模型，并使用独立的默认 dry-run 输出路径。

## 四类真实 Planner 请求

`multi_turn_base_66`、`multi_turn_miss_func_66`、`multi_turn_miss_param_66`、`multi_turn_long_context_66` 已分别通过官方 CLI 的真实模型请求、严格 Planner 解析和原工具范围校验，每条输出 4 个计划轮。调用使用 ChatGPT 登录，不使用项目中的 LLM API/SDK 后端。

证据位于 `/mnt/data/toolweave_planner_v2_audit/real_planner_results.json`、`real_planner_examples/` 和 `codex_calls/`。这四次测试发生在最终补充“生成器机制放 reason”与参数预算之前；原请求与返回保留原样。它们不证明最终版本四类完整数据链均能通过验收。

## `multi_turn_base_33` 的完整实测与修正

使用真实 update-2、K=16 的边界 seed，均值进度 0.625、边界分数 0.9375。初始配置恢复为原训练 parquet 中的完整配置，保留 `project`、`backup_scripts` 顺序。项目 pipeline 生成参数、执行 VM、逐轮生成和验证 Query、整段润色并验收；没有手工改写模型输出，也没有将任务执行交给 Codex。

首次真实 Planner 输出的 4 轮为 `ls` → `cat/wc/grep` → `get_user_id/send_message` → `view_messages_sent`。这次实测揭示并保留了两项结果：

1. 现有 `action_minimality_gate` 没识别用户明确的 “measure its line count”，误将 `wc` 归为额外调用。修正后，计数意图必须与 wc 的 line/word/character 模式一致；只读文件、只查匹配行或单位不匹配的反例仍被拒绝。
2. 生成的消息有 202 字符，超过现有 200 字符上限，正确被 `parameter_complexity_gate` 拒绝。保留此限制，并将同一预算交给 Planner、Parameter Generator，提前在 VM 调用前校验；没有放宽 gate 来接受这条消息。

这两个结果保存在 `/mnt/data/toolweave_planner_v2_audit/full_base33/`，包括修正前结果和完整模型请求。这条早期样本不能被报告为最终成功候选。

最后版本的实测输出位于 `/mnt/data/toolweave_planner_v2_audit/full_base33_current/`。还由这条真实任务补齐了 `wc(mode="c")` 对“Count the characters”的参数依据映射，并限定三种单位别名只用于 wc。

最后 Judge 首次拒绝了“删除最新消息”：返回文案写的是 first message，但 BFCL 实现倒序查找，实际 pre/post 显示本任务的临时消息被删除、旧消息保留。原 Judge 输入只有简化返回和工具名称。现在 Judge/Refine 接收实际使用的工具 schema 与完整状态证据，同时明确这些内部快照不是 Actor 的可见历史。没有改写 BFCL 返回，也没有改动原 Query 或 GT 来规避拒绝。

## 前一轮完成记录（单条 base）

- 回归测试：**187 passed，1 deselected**。未运行项是上述缺少历史文件的测试。多进程队列测试产生 4 条 Python fork 弃用提示，测试通过。
- 当前输入 schema：400 条任务全部通过，包括最后的参数预算字段；导出模型调用数为 0。
- 当前真实样本：`multi_turn_base_33` 最终 **SUCCEEDED**，1 次构造尝试、1 个 Planner 计划、5 个最终轮、9 个实际函数调用。
- 所有验收 gate、新 VM 重放、最终语义 verifier、最终 Quality Judge、候选 JSON schema 与 Training 实际验证器通过。最终消息为 143 和 159 字符，均满足叙事的 160 字符预算及项目的 200 字符上限。
- 实测模型返回全部来自官方 Codex CLI。中断恢复和检查器修正后的重放，只复用角色、提示词完全相同且已完成的真实返回；BFCL VM 从初始状态重新执行。参数、Query、Planner 输出与 GT 没有人工修改。

最终完整记录为 `full_base33_current/result.json`；`candidate.json` 是该结果中的接受候选，`training_sample.json` 是其中的训练样本。之前拒绝的结果和请求仍在同一审计目录或早期 `full_base33/` 目录内保留，不能用最终成功覆盖早期失败事实。

上述阶段只对一个最终真实 base 样本完成端到端验收。随后完成的四类实测见下节；批量成功率仍未测量。

## 2026-10-07：四类完整真实模型实测

当前版本用实际训练 parquet 中的四条 `_33` 原始任务完成参数生成、真实 BFCL VM 执行、逐轮 Query/验证、整段润色、类别转换、最终语义验证、独立新 VM 重放和最终 Judge。四例均产生 Candidate，并通过 Candidate JSON schema 与 Training 实际验证器。

| 类别 | 实际源 ID | 最终轮数 | 最终 GT 调用数 | 结果 |
| --- | --- | ---: | ---: | --- |
| base | `multi_turn_base_33` | 4 | 7 | SUCCEEDED |
| miss_func | `multi_turn_miss_func_33` | 5 | 9 | SUCCEEDED |
| miss_param | `multi_turn_miss_param_33` | 5 | 7 | SUCCEEDED |
| long_context | `multi_turn_long_context_33` | 3 | 6 | SUCCEEDED |

每例通过全部 11 项验收（包含模型最终语义检查）；最终运行每例 1 次构造尝试、1 个 Planner 计划，均未使用 Query Refine。这是修复后的一次最终运行，不能抹去此前失败，也不能作为未经筛选的 100% 接受率。

- Missing Function 隐藏 `delete_message`；0-based turn 3 的 GT/执行记录为空，turn 4 恢复完整 schema 后执行删除和检查。其他初始工具不能完成撤回。
- Missing Parameter 的必要值为 `get_user_id.user`。此前列出可选用户并不等于选定接收者；turn 2 的请求省略选择，GT/执行记录为空，turn 3 由用户补充 Alice 后执行。
- Long Context 在 turn 0 的实际 `cat` 返回暴露 3369 JSON 字符；turn 1 从该返回选取函数名和最后一条部署更新指令用于消息。后轮不重新读取/搜索脚本，也不直接在 Query 中给出完整答案。最终语义 verifier 与 Judge 均确认实际历史使用。2048 字符阈值仍是项目政策。

实测暴露并修正了最终 verifier 缺少类别/完整工具可见性、Python 声明模式的依据误拒绝、逐轮 verifier 缺少删除前后效果证据，以及 Judge 把用户源文件业务函数名当成工具 API 的误拒绝。LC 曾产生仅有大返回但绕过历史的样本，该例正确被拒绝；现已加强 Planner 类别规则和 LC Query 的实际历史输入。旧拒绝和中断结果原样保留。

Judge/Refine 的新解释是明确标记的项目补充，不能宣称仍是未修改的论文提示词。实际源 VM 的 `send_message` 返回 `message_id` 字典，源 schema 声明整数；该差异没有被修复或隐藏，记录在返回契约诊断中。这四例没有按整数消费该返回值；输入参数、任务效果与依赖仍必须通过验收。

官方 CLI 曾因 usage limit 中断。现在额度错误单独延期，保留 checkpoint、释放队列 claim 为 PENDING，不累计构造失败或写 terminal drop；额度恢复后重启，未改用 API。

最终运行目录为 `/mnt/data/toolweave_four_category_smoke/`，run ID 为 `run_20261007T040155_092238Z`。完整报告 `验证报告.md` 包含每例最终 Query/GT、Planner 输入/提示词/原始返回链接及验证原文；机器汇总为 `validation_summary.json`。各例的同名 run 子目录保存 `candidate.json`、`training_sample.json`、`validation_manifest.json`、`vm_trace.jsonl` 和三个 VM 初始快照；共享 `codex_calls/` 保存真实角色请求、返回和事件流。Generator 的 82 个 Python/prompt 文件哈希与最终运行 manifest 一致。

最后运行包含 19 次新的真实 CLI 请求及 58 条完全相同角色/提示词的已完成真实返回复用。测试缓存只存在于 smoke 脚本，不属于生产后端；每次运行 VM 都重新创建，四例合计实际执行 29 个合成调用和 29 个最终 GT 重放调用。没有人工修改模型输出、参数、Query、GT 或 Observation。

这些 smoke seed 的原任务内容和环境是真实数据，进度 0.5、边界分数 1、epoch/step 0 是明确标记的占位，不代表这四例经过真实 RL 边界测量。与前一节真实 update-2 边界 seed 的来源不同。

当前回归为 **216 passed，1 deselected**，JUnit 为 `/mnt/data/toolweave_four_category_smoke/pytest_results.xml`。唯一未运行项仍缺历史候选文件；4 条 fork 弃用提示，0 失败。400 条最新 Planner 输入和实际初始快照已重新导出并通过 schema，模型调用数为 0。相关回归主要使用脚本化模型返回和真实 VM；四例真实模型证据单独记录。

复跑命令及测试的客观范围见 [FOUR_CATEGORY_SMOKE_20261007.md](FOUR_CATEGORY_SMOKE_20261007.md)。当时没有发布生产候选、启动训练、提交或推送；后续公开代码与样例的范围见本文开头的数据包链接。

## 如何解释结果

四类各一个真实候选通过，证明四类流程在这些实例上可以执行并通过当前验收。四例属于相关的同一 `_33` 场景，不能证明系统完美、全部跨轮依赖已确定性恢复、Codex 相对 Gemma 的质量优越性或最终训练收益。

当前 LC 的 2048 字符暴露阈值、部分确定性语言意图识别和模型语义 verifier 都有适用范围。四类批量端到端接受率、人工误接收/误拒绝率、耗时和训练收益尚未测量。
