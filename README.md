# Lightweight Coding Agent（v0.2）

v0.2 在 v0.1 的通义千问调用能力之外，新增了一个独立的 Local Tool Layer，用于在项目的 `workspace/` 目录内安全地执行有限的文件操作和本地命令。

当前版本仍然不是 Agent：工具层尚未接入 LLM，不包含 Tool Calling、Agent Loop、Planning、Reflection、Context Manager、Verification、Patch Editing、多 Agent 或 Git 自动修改。

## 环境要求

- Conda
- Python 3.10 或更高版本
- 阿里云百炼 API Key（仅运行 v0.1 模型调用 CLI 时需要）

## 安装

创建并激活 Conda 环境：

```bash
conda create --name coding-agent python=3.10
conda activate coding-agent
python -m pip install -r requirements.txt
```

## 模型配置与 CLI

复制 `.env.example` 为 `.env`，然后填写 API Key：

```text
DASHSCOPE_API_KEY=你的_API_Key
LLM_BASE_URL=
LLM_MODEL=
```

`LLM_BASE_URL` 和 `LLM_MODEL` 留空时，分别使用：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `qwen-plus`

运行模型调用 CLI：

```bash
python main.py "你好，请介绍一下你自己"
```

不要提交包含真实密钥的 `.env` 文件。

## Local Tool Layer

`tools.py` 提供 5 个工具：

- `list_files`：列出目录，并区分文件、目录、符号链接和其他条目。
- `search_files`：递归搜索常见 UTF-8 代码及文本文件。
- `read_file`：读取带长度限制的 UTF-8 普通文件。
- `write_file`：写入 UTF-8 普通文件，并按需创建父目录。
- `execute_command`：验证并执行严格白名单中的本地命令。

所有工具只访问 `tools.py` 同级的 `workspace/` 目录，并统一返回：

```python
ToolResult(success: bool, output: str = "", error: str | None = None)
```

可以通过注册表统一分发：

```python
from tools import execute_tool

result = execute_tool("read_file", {"path": "README.md"})
```

## 命令白名单

v0.2 只接受以下参数列表：

```text
["python", "<workspace 内的脚本.py>"]
["python", "-m", "pytest"]
["pytest"]
["git", "status"]
["git", "diff"]
```

命令不会经过 shell；不支持管道、重定向、命令拼接或额外参数。`python` 和 `pytest` 使用当前虚拟环境的 Python 解释器执行。

## 安全边界

- 工具路径必须是 workspace 内的相对路径。
- 拒绝绝对路径、`..`、Windows 保留路径和路径中的符号链接。
- 解析后的路径必须仍位于 workspace 内。
- Agent 项目源码位于 workspace 之外，文件类工具不能直接读取或修改它。
- 搜索跳过 `.git`、`.venv`、`venv`、`node_modules`、`__pycache__` 和非 UTF-8 文件。
- 文件内容、搜索结果、命令输出和错误信息均有限长。
- subprocess 固定使用 workspace 作为 `cwd`，设置 30 秒超时，并明确使用 `shell=False`。
- 命令策略与 subprocess 执行分别位于 `tools.py` 和 `local_environment.py`。
- `git status/diff` 要求 `workspace/.git` 是真实目录，并显式绑定该 Git 目录和工作树，不会向上使用 Agent 自身仓库。

这是一层进程内路径与命令策略，不是操作系统沙箱。被允许执行的 Python 脚本和 pytest 测试本身仍可运行任意 Python 代码，也可能主动访问 workspace 外的文件，包括 Agent 项目源码。因此 workspace 中的可执行代码必须视为可信代码；若需要强进程隔离，应在后续版本引入容器、受限系统用户或其他操作系统级沙箱。

`workspace/` 的运行内容默认被外层 Agent 仓库忽略；`.gitkeep` 仅用于在克隆 Agent 项目后保留空目录。如果需要使用 `git status` 或 `git diff`，应将目标 Git 仓库克隆到 `workspace/`，或在该目录中初始化独立仓库。

## 自动创建父目录

`write_file` 会自动创建目标文件的父目录，使后续 Coding Agent 可以创建新的模块或目录结构，而不需要额外开放一个目录创建工具。创建前后都会重新进行 workspace 和符号链接检查。

## 测试

```bash
python -m pytest -q
```

测试使用 pytest 的 `tmp_path`，不会读写真实项目 workspace。覆盖正常文件操作、路径越界、路径穿越、符号链接、UTF-8 错误、输出截断、命令白名单、非零退出、超时和注册表异常隔离。

## 文件职责

- `config.py`：读取和校验模型环境变量。
- `llm.py`：封装普通 Chat Completions 调用。
- `main.py`：提供单轮模型调用 CLI。
- `tools.py`：定义统一工具结果、路径与命令安全策略、5 个工具和工具注册表。
- `local_environment.py`：执行已经通过校验的 subprocess 参数列表并返回原始执行结果。
- `tests/test_tools.py`：使用临时 workspace 验证 Local Tool Layer。
