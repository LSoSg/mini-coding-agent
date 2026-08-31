# Lightweight Coding Agent（v0.4）

v0.4 在 v0.3 的 Tool-Calling Agent Loop 前加入强制结构化规划，并用计划约束每一次本地工具调用。

```text
用户任务
  → 受控 workspace 根目录清单
  → 无工具的结构化规划
  → 计划匹配器
  → Local Tool Layer
  → Observation / 有限重规划
  → 计划完成 + 当前 revision 验证成功
  → 最终回答
```

Agent 只能访问项目下的 `workspace/`。规划层不替代 v0.2 的路径与命令安全检查；工具调用的完整链路是：

```text
LLM Tool Call
  → Agent 计划匹配
  → execute_tool / TOOL_REGISTRY
  → Tool Layer 路径与命令安全检查
  → LocalEnvironment
  → subprocess（shell=False）
```

## 安装

推荐使用 Conda 创建独立环境：

```bash
conda create --name coding-agent python=3.10
conda activate coding-agent
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并填写百炼 API Key：

```text
DASHSCOPE_API_KEY=你的_API_Key
LLM_BASE_URL=
LLM_MODEL=
```

后两项留空时使用默认值：

- Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- Model：`qwen-plus`

不要提交包含真实密钥的 `.env`。

如果 Windows/Conda 中曾设置无效的 `SSL_CERT_FILE`，OpenAI 客户端初始化可能报 `FileNotFoundError`。请删除该无效环境变量，或将它改为当前 Conda 环境中真实存在的 CA 文件路径；不要在代码中关闭 SSL 校验。

## 运行

先把待处理代码放入 `workspace/`，再运行：

```bash
python main.py "Inspect the workspace and tell me what this project does."
```

代码创建任务示例：

```bash
python main.py "请实现标准的 Dijkstra 最短路径算法，并编写测试进行验证。"
```

CLI 会输出根目录预检、初始计划、计划偏离、重规划、工具结果、验证证据、最终状态以及完整 Plan History。`write_file.content` 不会出现在计划或参数日志中。

## 结构化计划协议

规划和执行使用同一个 LLM，但规划请求不携带 `tools`。模型必须返回严格 JSON：

```json
{
  "goal": "实现标准 Dijkstra 算法并验证",
  "success_criteria": [
    "生成 dijkstra.py",
    "测试通过"
  ],
  "steps": [
    {
      "id": "step_1",
      "description": "创建算法实现",
      "tool": "write_file",
      "argument_constraints": {"path": "dijkstra.py"},
      "rationale": "这是用户要求的目标文件"
    },
    {
      "id": "step_2",
      "description": "创建测试",
      "tool": "write_file",
      "argument_constraints": {"path": "test_dijkstra.py"},
      "rationale": "提供可执行的正确性验证"
    },
    {
      "id": "step_3",
      "description": "运行完整测试",
      "tool": "execute_command",
      "argument_constraints": {"command": ["python", "-m", "pytest"]},
      "rationale": "获得真实验证证据"
    }
  ]
}
```

程序会校验字段、步骤 ID、工具名、真实工具参数、workspace 相对路径、命令白名单、步骤上限和验证顺序。`write_file` 计划只保存 `path`，完整 `content` 在执行时生成，不参与计划匹配。

默认边界：

- 最多 2 次计划生成/纠错尝试；
- 最多接受 2 次重规划；
- 每份计划最多 12 个步骤；
- 模型提前返回 final 时最多提醒 3 次；
- Agent Loop 最多 20 步；
- 所有任务都先规划，纯信息任务可以生成零工具步骤计划。

## 计划约束与重规划

执行时只允许匹配当前 pending step 的工具调用：工具名必须相同，`path`、`keyword`、`command` 等约束必须匹配。默认参数会在匹配前规范化；`write_file.content` 例外。

一个 assistant response 包含多个 Tool Calls 时，它们必须依次匹配连续计划步骤。任意一个调用偏离，整批都不会执行，每个 `tool_call_id` 都会收到结构化失败 observation，然后进入无工具的重规划调用。

工具成功后对应步骤才会变为 `COMPLETED`。工具失败会作为 observation 返回模型，步骤保持 `PENDING`，允许模型修正并重试。历史计划及其完成状态不会被覆盖。

## Verification-aware termination

`COMPLETED` 同时要求：

- 当前计划的所有步骤成功完成；
- 模型返回无 Tool Call 的最终文本；
- 需要验证的任务已有成功验证证据；
- 最近验证对应当前 workspace revision。

每次成功 `write_file` 都会增加 workspace revision。验证成功后再次写入文件，会使旧验证失效，必须重新验证。以下命令可产生验证证据：

```text
["pytest"]
["python", "-m", "pytest"]
["python", "<workspace 内脚本.py>"]
```

`git status` 和 `git diff` 只是信息命令，不构成验证证据。模型文字中的“测试通过”也不构成证据。

最终状态包括：

- `COMPLETED`：计划完成，并满足必要的真实验证条件；
- `PLAN_FAILED`：无法获得合法计划、超过重规划次数，或反复跳过未完成计划；
- `VERIFICATION_REQUIRED`：当前 workspace revision 缺少成功验证；
- `MAX_STEPS_REACHED`：达到 Agent Loop 步数上限；
- `FATAL_ERROR`：LLM API 或消息协议出现无法继续的错误。

计划 JSON 错误是可恢复的规划错误，不会直接归类为 `FATAL_ERROR`。

## Local Tool Layer 安全边界

- 文件工具只接受 `workspace/` 内的相对路径；
- 拒绝绝对路径、`..`、保留系统路径和路径中的符号链接；
- 搜索跳过依赖、Git、缓存目录和非 UTF-8 文件；
- 文件、搜索、命令和错误输出均限制长度；
- 命令以参数列表执行，固定 workspace `cwd`、30 秒超时、`shell=False`；
- 只允许 `python <script.py>`、`pytest`、`git status` 和 `git diff` 的既定形式；
- Git 命令要求 workspace 拥有自己的 `.git`，不会向上使用 Agent 项目的仓库。

这不是操作系统级沙箱。被允许的 Python 脚本和 pytest 测试本身可以运行任意 Python 代码，因此只应在 workspace 中放置可信代码。更强隔离需在后续版本引入容器或受限系统账户。

## 测试

```bash
python -m pytest -q
```

该命令按 `pytest.ini` 收集 `tests/` 下的测试，包括 `tests/test_agent.py`、`tests/test_planning.py`、LLM/Schema 测试和原有 Tool Layer 测试。测试使用 Fake LLM，不调用真实百炼 API；Tool Layer 测试使用临时目录，不修改真实 workspace。

如果 Windows 上遇到旧 pytest 缓存目录的 `WinError 5`，可执行：

```bash
mkdir -p tmp
python -m pytest -q --basetemp=tmp/pytest-user -p no:cacheprovider
```

## Dijkstra 回归验收

在 workspace 中保留无关的 `calculator.py` 和 `test_calculator.py`，然后运行 Dijkstra 创建任务。期望轨迹接近：

```text
root inventory
→ explicit plan
→ write dijkstra.py
→ write test_dijkstra.py
→ python -m pytest
→ COMPLETED
```

验收重点：

- 不读取 `calculator.py` 或 `test_calculator.py`；
- 不尝试白名单之外的 pytest 参数；
- 每个动作都能映射到一个计划步骤；
- 最终成功同时具有完整计划和真实验证证据。

## 文件职责

- `config.py`：加载并校验模型配置；
- `llm.py`：发送 Chat Completions；无 tools 时返回文本，有 tools 时返回 assistant message；
- `planning.py`：计划数据模型、严格 JSON 校验、参数规范化和计划匹配；
- `agent.py`：规划、计划约束 Agent Loop、重规划、验证证据和终止状态；
- `tool_schemas.py`：暴露给 LLM 的 OpenAI-compatible Tool Schema；
- `tools.py`：统一 `ToolResult`、工具安全策略、实现与 `TOOL_REGISTRY`；
- `local_environment.py`：执行已通过安全校验的 subprocess 参数列表；
- `main.py`：处理 CLI 输入并展示状态、计划历史和结果；
- `tests/test_planning.py`：结构化计划协议与匹配器测试；
- `tests/test_agent.py`：Fake LLM 规划、执行、重规划和终止测试；
- `tests/test_tools.py`：v0.2 Local Tool Layer 安全与功能回归测试。

## v0.4 明确不包含

Reflection、Working Memory、Context trimming、摘要、Patch/AST Editing、通用 Replanning 框架、Planner Agent、第二个模型、trajectory 文件日志、token/cost 统计、Human confirmation、多 Agent、RAG、Web UI、Docker sandbox、自动 Git commit 或 push。
