# Mini Coding Agent

一个从零实现的轻量级 Coding Agent 。项目通过阿里云百炼 OpenAI-compatible API 调用 Qwen Builder 和 DeepSeek Verifier。

## Git 仓库

https://github.com/LSoSg/mini-coding-agent.git

## 快速开始

```bash
conda create -n coding-agent python=3.10
conda activate coding-agent
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并填写 API Key：

```text
DASHSCOPE_API_KEY=你的_API_Key
LLM_BASE_URL=
BUILDER_MODEL=qwen-plus
VERIFIER_MODEL=deepseek-v4-flash
```

将待处理项目放入 `workspace/`，然后在本仓库根目录运行：

```bash
python main.py "检查项目，修复问题并说明修改内容"
```

## 工作流

```text
Workspace 快照与原测试发现
→ Qwen 生成结构化计划
→ 按计划执行受限本地工具
→ DeepSeek 独立审查并给出建议
→ 在临时副本中恢复原测试并回归
→ 输出最终状态与完整轨迹
```

## 特色功能

- **安全工具层**：支持文件查看、搜索、读写和受限命令执行。所有文件操作限制在 `workspace/`，命令使用白名单、固定 cwd、30 秒超时和 `shell=False`。
- **结构化规划**：本地操作前先生成 JSON 计划，校验工具、参数和路径。Tool Call 必须匹配当前步骤，偏离时整批拒绝并进行有限重规划。Builder 不被强制创建测试或生成验证计划。
- **Working Memory**：在单次任务中维护目标、成功标准、显式约束、计划进度、workspace revision 以及已读取和已修改文件，每轮注入 Builder，减少目标丢失和重复读取。
- **异构双模型**：Qwen 负责规划和代码修改；DeepSeek 在隔离上下文中审查最终源码、寻找反例和未确认假设，降低同一模型“自己出题、自己判卷”的自洽偏差。
- **Verifier 建议机制**：Verifier 返回严格 JSON，但 `PASS/FAIL` 仅作为建议展示，不直接决定任务成败；API 或响应协议异常仍会返回 `VERIFIER_FAILED`。
- **快照原测试回归**：任务开始时复制 workspace 并记录原测试。Agent 结束后，在临时副本中恢复原测试和 pytest 配置，再针对最终代码执行，避免通过修改或删除测试获得成功状态。

## 当前限制

- **正确性仍是核心挑战**：测试通过只能提供有限证据，无法证明代码在所有输入和真实场景下都满足需求。
- **工程能力仍有限**：目前主要面向 Python 项目和有限命令集合，尚未支持插件化工具、更多语言/构建系统、IDE/LSP、Git 工作流等。后续可以通过统一 Tool 接口扩展不同语言工具链，而不改变 Agent 主循环。
- **Working Memory 是轻量状态而非长期记忆**：当前只服务于单次任务，不做跨任务学习、向量检索或持久化记忆。

## 开发思考

这个项目不是一次性设计出来的，而是从最小 LLM 调用开始逐步演化：Local Tool Layer → Tool Calling → Verification → Planning → 原测试回归 → Working Memory → 独立 Verifier。每一层基本都来自前一版本真实运行中暴露的问题。

开发过程中一个很明显的体会是：Coding Agent 很容易不断“做加法”。出现失败时，可以继续增加 Planner、Memory、Verifier、状态和规则，但模块越多，状态之间的耦合也越复杂；在 Vibe Coding 中，这一点尤其明显——代码增长很快，但一旦行为异常，很难判断问题来自 Prompt、模型决策、计划状态、工具执行还是验证逻辑。
