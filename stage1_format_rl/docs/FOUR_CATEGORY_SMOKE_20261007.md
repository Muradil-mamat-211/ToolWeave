# 四类完整真实模型 smoke：2026-10-07（早期历史版本）

本文记录 `run_20261007T040155_092238Z`，不是后来发布的四份候选。公开样例来自同日后续的 `run_20261007T140417_209480Z`，两次运行的 Query、GT 和验收数量不同。当前方法与公开样例见 [Codex 构建指南](../../docs/codex-data-synthesis.md) 和 [数据包](../../data/codex-synthesis/README.md)。以下保留早期运行的原始结果与局限。

较早版本已对实际训练 parquet 中四条 `_33` 源任务完成完整 Generator 流程，模型返回来自官方 `codex exec` 与已有 ChatGPT 登录。没有项目直接 LLM API/SDK 调用。实际工具执行由 Python BFCL VM 完成。

严格复审随后发现精确文本标签和 LC 信息分布的问题；新增策略重查这四条旧候选时全部拒绝。以下结果保留为原 11 项验收的历史，不作为新版质量通过的证据。修订和新的独立实测见 [QUERY_CONTRACT_HARDENING.md](QUERY_CONTRACT_HARDENING.md)。

| 类别 | 源 ID | 最终轮数 | 最终 GT 调用数 | 最终结果 |
| --- | --- | ---: | ---: | --- |
| base | `multi_turn_base_33` | 4 | 7 | SUCCEEDED |
| miss_func | `multi_turn_miss_func_33` | 5 | 9 | SUCCEEDED |
| miss_param | `multi_turn_miss_param_33` | 5 | 7 | SUCCEEDED |
| long_context | `multi_turn_long_context_33` | 3 | 6 | SUCCEEDED |

每例通过 11 项验收（含模型最终语义检查）、独立新 VM 执行全部最终 GT、Quality Judge、Candidate JSON schema 和 Training 实际验证器。最后运行每例 1 个 Planner 计划、1 次构造尝试、未使用 Query Refine。前面的排错失败保留，因此这些数字不是未经筛选的生成成功率。

## 实际类别行为

- **Base**：列目录，检查脚本的函数声明、行数、字符数，查 Catherine 后发送一条摘要，查看发件历史。GT 包含 7 次调用。
- **Missing Function**：生成临时部署摘要并确认，要求撤回时缺少 `delete_message`，受影响轮 `GT=[]`；下一轮恢复 schema 后删除刚发送的摘要并确认，旧消息保留。缺失/恢复 turn_id 为 3/4，按 0 开始。
- **Missing Parameter**：先检查脚本和列用户，之后请求发送但没有选择接收者，受影响轮 `GT=[]`；用户明确补充 Alice 后查 ID、发送并查看历史。缺失/恢复 turn_id 为 2/3。列出候选名字不等于选定收件人。
- **Long Context**：读取实际扩展文件，`cat` 返回的 JSON 长度为 3369 字符。后轮从此前内容提取函数名和最后一条部署更新指令来发送消息；既不重新读取/搜索，也不把完整答案直接提供在后轮 Query 中。最终 verifier 和 Judge 接受该实际历史依赖。

LC 最后发出的真实消息为：

```text
Function: deploy; last deployment update before final checks: # update the server
```

这是 81 字符，符合 200 字符参数预算。LC 的 2048 字符暴露门槛是项目政策，不是 BFCL 官方标准或模型 token 数；长度不替代后轮信息使用检查。

## 运行与复跑

在仓库根目录，安装项目依赖并确认已有 CLI 的 ChatGPT 登录后：

```bash
export PYTHONPATH="$PWD/code/AWorld-RL-stage1-worktree/EnvTuning${PYTHONPATH:+:$PYTHONPATH}"
python stage1_format_rl/scripts/run_codex_category_smoke.py \
  --dataset /path/to/bfcl_train.parquet \
  --output /path/to/four_category_smoke \
  --codex-binary /path/to/codex \
  --concurrency 2
```

默认选择四条 `_33`；`--sample-ids` 可替换源 ID。输入不含原学生 rollout 或拓扑 JSON；原工具定义提供一份，并保留原工具恢复事件。每个角色是独立 ephemeral CLI 会话，Python 明确传递后续所需字段。没有手工修改模型输出、参数、Query、GT 或 VM Observation。

`--reuse-completed` 可复用同一输出目录下角色与提示词完全相同、已完成且没有禁止工具事件的真实旧返回。该测试缓存不属于生产 CLI 后端，VM 每次仍重新创建并执行。最后一次实测包含 19 次新的真实 CLI 请求和 58 条真实完成返回复用，实际执行 29 个合成调用和 29 个新 VM 重放调用。

源任务、GT、工具和初始配置来自真实 parquet，SHA256 为 `f6c9311b3b7977b307fcb8d175603e828dbca0542bb3abca4f4bdf922bc8f293`。完整文件系统根顺序为 `project`、`backup_scripts`。Smoke 的进度 0.5、边界分数 1、epoch/step 0 是显式占位，不能当成观测到的 RL 边界。

## 记录位置

本次工作区审计根目录为 `/mnt/data/toolweave_four_category_smoke/`，最终 run ID 为 `run_20261007T040155_092238Z`。

| 文件 | 内容 |
| --- | --- |
| `验证报告.md` / `validation_summary.json` | 全部实际 Query/GT、逐例结果、原始证据链接与限制 |
| `run_20261007T040155_092238Z_manifest.json` | 源数据哈希、82 个 Generator Python/prompt 文件哈希、运行设置 |
| `run_20261007T040155_092238Z_results.json` | 最终四例状态、CLI 请求与复用计数 |
| `<sample_id>/<run_id>/seed.json` | 原任务和明确标记的 smoke 占位进度 |
| 同目录 `planner_input_attempt_1.json` / `planner_prompt_attempt_1.txt` | 完整实际 Planner 输入和提示词 |
| 同目录 `vm_trace.jsonl` / `vm_sessions/` | 每次真实执行的返回、pre/post，以及三个 VM 实例的初始状态 |
| 同目录 `candidate.json` / `training_sample.json` | 验收通过的候选与实际训练格式样本 |
| 同目录 `validation_manifest.json` | 11 项验收、Judge 原文、schema 和 Training 验证结果 |
| `codex_calls/` | 每个角色实际请求、原始返回、事件流与 stderr |
| `pytest_results.xml` | 216 个通过的回归测试结果 |

Missing Function 的恢复轮在训练 `question` 中表示为 `[]`；其完整恢复 schema 和继续请求实际存放在 `processed_question`，遵循已有 Training 协议。

## 修复与客观限制

实测修复了最终 verifier 的类别/工具可见性输入、Python 声明模式依据误拒绝、逐轮删除效果的证据不足、Judge 对用户业务函数名的接口泄露误判和 LC 历史绕过。官方 CLI usage limit 现在使队列任务延期并回到 PENDING，不当作构造失败，不转用 API。具体实现见 [Planner 设计](PLANNER_V2_DESIGN.md) 与 [完整验证历史](PLANNER_V2_VALIDATION.md)。

Judge/Refine 使用明确标记的项目补充，不能宣称是未修改的论文提示词。源 BFCL 的 `send_message.message_id` 实际为字典、返回 schema 声明整数，这一差异仍存在且有诊断记录。本次任务不依赖该整数声明；真实效果、输入参数和后续依赖仍必须通过验收。没有修改源 VM 返回来消除差异。

当前回归为 **216 passed，1 deselected**，4 条 fork 弃用提示，0 失败。唯一未运行的历史恢复测试缺少其实际历史候选文件，没有伪造替代。400 条最新 Planner 输入已重新导出并通过 schema，模型调用数 0。回归模型多数是 fixture；四例真实模型测试单独记录。

四例来自相关的同一 `_33` 场景，证明这四种流程在这些实例上可行。尚未测量不同工具和任务的大批量成功率、人工误接收/误拒绝率、Codex 对 Gemma 的优劣或训练收益。模型 verifier/Judge 不是真值证明。当时的代码与候选仅保存在本地，没有提交、推送、发布到生产队列或启动训练；后续公开的代码与候选见本文开头的链接。
