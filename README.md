# Lightweight Coding Agent（v0.3）

v0.3 将阿里云百炼 OpenAI-compatible LLM Client 与本地 Tool Layer 连接为一个最小 Tool-Calling Coding Agent，并使用真实命令结果实现 Verification-aware termination。

```text
User Task → LLM → Tool Call → Safe Local Tool → Observation → LLM
                                                       ↓
                                      Verification Evidence → Final Status
```

## 当前能力

- 通过百炼调用通义千问 Chat Completions。
- 维护标准 assistant/tool message history。
- 支持 `list_files`、`search_files`、`read_file`、`write_file` 和 `execute_command`。
- 顺序执行单轮响应中的全部 Tool Calls。
- 将统一 `ToolResult` 序列化为 JSON observation。
- 记录真实验证证据，拒绝无验证或验证失败后的虚假完成声明。
- 使用 `MAX_STEPS` 和 `MAX_VERIFICATION_REQUESTS` 防止无限循环。

## 安装

```bash
conda create --name coding-agent python=3.10
conda activate coding-agent
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填写百炼 API Key：

```text
DASHSCOPE_API_KEY=你的_API_Key
LLM_BASE_URL=
LLM_MODEL=
```

后两项留空时默认使用：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `qwen-plus`

不要提交包含真实密钥的 `.env`。

## 运行 Agent

将待处理代码放入 `workspace/`，然后运行：

```bash
python main.py "Inspect the workspace and tell me what this project does."
```

代码修改任务示例：

```bash
python main.py "The tests are failing. Find the bug and fix it without modifying the tests."
```

CLI 会显示每个 Agent step、工具名称、隐藏具体写入内容后的参数摘要、工具成功状态、验证证据和最终状态。可能的状态包括：

- `COMPLETED`：信息任务已回答，或当前代码已有成功验证证据。
- `VERIFICATION_REQUIRED`：代码任务没有获得当前有效的成功验证。
- `MAX_STEPS_REACHED`：达到 Agent step 上限。
- `FATAL_ERROR`：LLM、配置或消息协议发生无法继续的错误。

## Tool Schema 与本地注册表

`tool_schemas.py` 中的 `TOOLS` 是 OpenAI-compatible JSON Schema，只负责告诉 LLM 可用工具及参数。

`tools.py` 中的 `TOOL_REGISTRY` 将工具名称映射到本地 Python 函数，只负责实际分发。模型生成的调用必须经过：

```text
LLM Tool Call
    → JSON arguments parser
    → execute_tool
    → Tool Layer 路径/命令安全检查
    → LocalEnvironment
```

Agent 不直接访问文件系统，也不直接调用 subprocess。

## Verification-aware termination

涉及创建、实现、修改、修复、重构、测试、构建或验证的任务需要验证。明确的查看、读取、列出、解释或描述任务不强制验证；模糊任务默认要求验证。

成功执行以下当前白名单命令可产生验证证据：

```text
["pytest"]
["python", "-m", "pytest"]
["python", "<workspace 内的脚本.py>"]
```

`git status` 和 `git diff` 是信息命令，不是验证证据。失败的测试会被记录为失败证据并返回模型，但不会立即终止 Agent。

每次成功 `write_file` 都会增加 workspace revision。成功验证只对当时 revision 有效；验证成功后再次写文件，必须重新验证。只有最后一条验证成功且对应当前 revision，代码任务才能返回 `COMPLETED`。

模型文字中的“测试通过”不构成验证证据。证据只能来自真实 `execute_tool("execute_command", ...)` 返回的 `ToolResult`。

## Local Tool Layer 安全边界

- 文件工具只接受 `workspace/` 内相对路径。
- 拒绝绝对路径、`..`、保留系统路径和路径中的符号链接。
- 搜索跳过依赖、Git、缓存目录和非 UTF-8 文件。
- 文件、搜索、命令和错误输出均有限长。
- 命令以参数列表执行，固定 workspace `cwd`，30 秒超时，`shell=False`。
- 只允许 `python <script.py>`、pytest、`git status` 和 `git diff`。
- Git 命令要求 workspace 拥有自己的 `.git`，不会向上使用 Agent 仓库。

这不是操作系统沙箱。被允许的 Python 脚本和 pytest 测试自身可以运行任意 Python 代码，甚至主动访问 workspace 外的文件，因此只应执行可信代码。强隔离需要后续引入容器或受限系统用户。

## 测试

```bash
python -m pytest -q
```

测试使用 Fake LLM，不调用真实百炼 API；Tool Layer 测试使用 `tmp_path`，不修改真实 workspace。

如果 Windows 上出现旧 pytest 缓存目录的 `WinError 5`，可以临时绕过该缓存：

```bash
mkdir -p tmp
python -m pytest -q --basetemp=tmp/pytest-user -p no:cacheprovider
```

## 手动 End-to-End：修复代码

在 `workspace/calculator.py` 中准备一个错误实现：

```python
def add(a, b):
    return a - b
```

在 `workspace/test_calculator.py` 中准备测试：

```python
from calculator import add


def test_add():
    assert add(2, 3) == 5
```

运行：

```bash
python main.py "The tests are failing. Find the bug and fix it without modifying the tests."
```

理想轨迹为读取文件、修复实现、执行 pytest、获得成功证据，然后返回 `COMPLETED`。

## 手动 End-to-End：从零创建代码

清空 workspace 中除 `.gitkeep` 外的任务文件，然后运行：

```bash
python main.py "Create calculator.py with add, subtract, multiply and divide functions. Also create pytest tests and run them to verify the implementation."
```

理想轨迹为创建实现和测试、执行 pytest、获得成功证据，然后返回 `COMPLETED`。

## 文件职责

- `config.py`：加载和校验模型配置。
- `llm.py`：发送 Chat Completions；普通模式返回文本，Tool Calling 模式返回 assistant message。
- `tool_schemas.py`：定义暴露给 LLM 的 OpenAI-compatible Tool Schema。
- `agent.py`：维护消息、执行 Agent Loop、记录验证证据并决定终止状态。
- `tools.py`：提供统一 `ToolResult`、工具安全策略、实际工具和注册表。
- `local_environment.py`：执行已经通过安全校验的 subprocess 参数列表。
- `main.py`：处理 CLI 输入并展示 Agent 返回的状态和结果。
- `pytest.ini`：限制 Agent 项目的测试发现范围，避免收集 workspace 和临时目录。
- `tests/test_agent.py`：Fake LLM Agent Loop 与验证终止测试。
- `tests/test_llm.py`：LLMClient 普通/Tool Calling 返回模式测试。
- `tests/test_tool_schemas.py`：Schema 与本地函数签名一致性测试。
- `tests/test_tools.py`：Local Tool Layer 安全和功能测试。

## v0.3 明确不包含

Planning、Reflection、Replanning、Working Memory、Context trimming、摘要、Patch/AST Editing、trajectory logger、token/cost 统计、Human confirmation、多 Agent、RAG、Web UI、Docker sandbox、额外验证 Agent、自动 Git commit 或 push。
