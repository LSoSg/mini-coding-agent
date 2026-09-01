# Mini Coding Agent

一个从零实现的轻量级 Coding Agent，通过阿里云百炼 OpenAI-compatible API 调用通义千问，不依赖 LangChain、LlamaIndex、OpenAI Agents SDK 等第三方 Agent 框架。项目核心逻辑均自行实现，包括 Tool Calling、对话历史、规划、工具执行、Working Memory、验证与终止条件。

## Git 仓库

https://github.com/LSoSg/mini-coding-agent.git

```bash
git clone https://github.com/LSoSg/mini-coding-agent.git
cd mini-coding-agent
```

## 如何运行

```bash
conda create -n coding-agent python=3.10
conda activate coding-agent
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`：

```text
DASHSCOPE_API_KEY=你的_API_Key
LLM_BASE_URL=
LLM_MODEL=
```

后两项留空时默认使用阿里云百炼兼容接口和 `qwen-plus`。请勿提交真实密钥。

将待处理项目放入 `workspace/` 后运行：

```bash
python main.py "检查项目，修复问题并运行测试"
```

运行项目自身测试：

```bash
python -m pytest -q
```

## 特色功能

- **安全工具层**：支持文件查看、搜索、读写和受限命令执行。所有文件操作限制在 `workspace/`，命令使用严格白名单、固定工作目录、超时控制和 `shell=False`。
- **结构化规划**：执行前先生成 JSON 计划，校验工具、参数、路径和验证步骤；实际 Tool Call 必须匹配当前计划，发生偏离时拒绝执行并进行有限重规划。
- **Working Memory**：在单次任务中显式维护目标、成功标准、全局约束、计划进度、文件 revision、已修改文件和验证状态，并在每轮模型调用前注入，减少长任务中的状态丢失与无效重复读取。
- **分层可信验证**：区分 Agent 自己运行的 `SELF` 验证和任务开始时原测试的 `ORIGINAL` 验证。Agent 完成后，在临时副本中恢复原测试并针对最终代码重新执行，避免仅通过修改测试来获得成功状态。
- **Verification-aware Termination**：模型文字中的“已完成”不直接作为成功依据；代码任务需要计划完成并获得与当前 workspace revision 对应的真实验证证据。
- **可观测执行过程**：CLI 展示 Planning、Tool Call、Working Memory、验证证据、计划历史及最终状态，便于分析 Agent 的决策和失败恢复过程。

本项目提供的是进程内工具与验证边界，并非操作系统级沙箱；被执行的 Python 代码应视为可信代码。