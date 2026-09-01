# Lightweight Coding Agent（v0.5）

v0.5 在 v0.4 的显式计划和安全 Tool Layer 之上，增加 workspace 快照与原测试回归验证。

它解决的问题是：

> Agent 自己编写或修改的测试通过，不等于实现真的正确。

Agent 仍然可以正常修改源码和测试，但最终验证会使用任务开始时快照中的原测试。修改断言、删除测试或调整 pytest 配置，不能让原测试回归得到通过。

```text
用户任务
  → 初始 workspace 快照与原测试发现
  → 显式计划
  → Agent 修改代码/测试
  → SELF 自验证
  → 模型 final
  → 临时副本恢复原测试
  → ORIGINAL 回归验证
  → 分层终止状态
```

## 安装

推荐使用 Conda：

```bash
conda create --name coding-agent python=3.10
conda activate coding-agent
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`：

```text
DASHSCOPE_API_KEY=你的_API_Key
LLM_BASE_URL=
LLM_MODEL=
```

后两项留空时默认使用：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `qwen-plus`

不要提交包含真实密钥的 `.env`。

如果 Windows/Conda 中的 `SSL_CERT_FILE` 指向不存在的文件，OpenAI 客户端可能在初始化时报 `FileNotFoundError`。请删除该无效环境变量，或将它改为当前 Conda 环境中真实存在的 CA 文件；不要关闭 SSL 校验。

## 运行

把待处理项目放入 `workspace/`：

```bash
python main.py "Fix the bug and verify the implementation."
```

v0.5 不要求固定项目布局，以下形式都可以：

```text
workspace/                 workspace/
├── app.py                 ├── src/
└── test_app.py            └── tests/
```

CLI 会显示快照状态、原测试发现结果、计划历史、自验证证据、原测试完整输出和最终验证等级。

## Workspace 快照

每次有效任务在第一次规划调用前都会保存 workspace 临时副本，并跳过：

- `.git`；
- `.venv`、`venv`；
- `node_modules`；
- `__pycache__`、`.pytest_cache`；
- `.pyc` 和 `.pyo` 文件。

快照不修改真实 workspace，并在 Agent 本次运行结束后清理。快照复制失败属于 `FATAL_ERROR`，不会退回无快照执行。

快照中会通过受限的 pytest `--collect-only` 发现任务开始时真实可收集的测试文件，因此支持根目录测试、`tests/` 布局和自定义 `python_files` 配置。

如果原项目存在导入错误或 pytest 配置错误，导致测试发现失败，Agent 仍可尝试修复，但最终最高只能达到 `SELF_VERIFIED`。

## SELF 与 ORIGINAL

### SELF

Agent 通过 `execute_command` 运行的 pytest 或 Python 脚本都属于 `SELF`：

- 测试可能由 Agent 自己生成；
- 已有测试可能已被 Agent 修改；
- 结果只能证明“当前实现和当前测试相互一致”。

### ORIGINAL

模型返回 final 后，外层程序创建一个新的临时验证副本：

1. 复制 Agent 最终 workspace；
2. 删除最终版本中的 pytest 配置与 `conftest.py`；
3. 恢复初始快照中的原测试、原配置、fixture 和 helper；
4. 显式运行任务开始时收集到的测试文件；
5. 不把结果反馈给模型，也不进入自动修复循环；
6. 验证完成后删除临时副本。

Agent 新增的测试不会进入 ORIGINAL 回归。对于常规 `tests/` 或 `test/` 目录，整个原测试目录都会恢复，以避免通过修改 helper 或 fixture 改变断言语义。

原测试输出会显示给终端用户，但限制为 20,000 字符，并将临时绝对路径替换为占位符。

## 终止状态

- `COMPLETED`：信息任务正常结束，或者最终代码通过 ORIGINAL 回归；
- `SELF_VERIFIED`：SELF 验证通过，但初始 workspace 没有可运行原测试，或原测试发现失败；
- `ORIGINAL_TESTS_FAILED`：SELF 验证通过，但初始快照中的原测试在最终代码上失败；
- `VERIFICATION_REQUIRED`：当前 workspace revision 缺少成功 SELF 验证；
- `PLAN_FAILED`：无法获得或遵循合法计划；
- `MAX_STEPS_REACHED`：达到 Agent Loop 步数上限；
- `FATAL_ERROR`：LLM、快照或外层 runner 出现无法继续的异常。

验证等级：

```text
SELF < ORIGINAL
```

CLI exit code：

- `COMPLETED`：0；
- `FATAL_ERROR`：1；
- 其他状态：2。

## 显式计划与安全工具层

所有任务仍然先生成严格 JSON 计划。规划请求不携带 tools；执行阶段只允许按顺序匹配当前计划步骤的 Tool Call，最多接受两次重规划。

工具调用仍经过：

```text
Agent plan matcher
  → execute_tool / TOOL_REGISTRY
  → 路径与命令安全检查
  → LocalEnvironment
  → subprocess（shell=False）
```

Agent 只能通过工具访问 `workspace/`，命令仍限定为 `python <script.py>`、pytest、`git status` 和 `git diff` 的既定形式。

## 测试

```bash
python -m pytest -q
```

测试使用 Fake LLM 和 `tmp_path`，不会调用真实百炼 API，也不会修改真实 workspace。覆盖内容包括：

- v0.2 Tool Layer 安全边界；
- v0.4 规划、重规划和 verification-aware termination；
- 快照创建、忽略规则和清理；
- pytest 原测试发现；
- 原断言、删除测试、pytest 配置、helper 和 conftest 恢复；
- SELF/ORIGINAL 证据及终止状态。

Windows 若遇到旧 pytest 缓存目录的 `WinError 5`，可运行：

```bash
mkdir -p tmp
python -m pytest -q --basetemp=tmp/pytest-user -p no:cacheprovider
```

## 文件职责

- `workspace_snapshot.py`：快照生命周期、原测试发现、验证副本和 ORIGINAL 回归；
- `pytest_snapshot_worker.py`：独立 pytest 子进程中的收集与执行入口；
- `agent.py`：规划、Agent Loop、SELF/ORIGINAL 证据和终止状态；
- `planning.py`：严格计划 JSON 校验和步骤匹配；
- `tools.py`：本地工具、安全策略和统一 `ToolResult`；
- `local_environment.py`：执行已通过校验的 subprocess 参数列表；
- `main.py`：CLI 和分层验证结果展示。

## v0.5 明确不包含

Hidden/Trusted Tests、测试文件写保护、逐工具完整性监控、容器、受限系统账户、Reflection、Working Memory、Patch Editing、多 Agent、自动修复循环、自动 Git commit 或 push。

v0.5 不是操作系统级沙箱。被允许执行的 workspace Python 代码仍能运行任意 Python 逻辑，因此只应处理可信代码。
