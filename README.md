# Lightweight Coding Agent（v0.1）

这是项目的第一个最小版本。目前只用于验证：命令行输入可以通过阿里云百炼的 OpenAI 兼容接口发送给通义千问，并在终端输出模型回复。

当前版本不包含 Tool Calling、文件或 Shell 操作、Agent Loop、Planning、Reflection、Memory、Verification、多 Agent 或 Web UI。

## 环境要求

- Python 3.10 或更高版本
- 阿里云百炼 API Key

## 安装

使用 Conda 创建名为 `coding-agent` 的 Python 3.10 虚拟环境：

```bash
conda create --name coding-agent python=3.10
```

激活环境并安装依赖：

```bash
conda activate coding-agent
python -m pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env`，然后填写 API Key：

```text
DASHSCOPE_API_KEY=你的_API_Key
LLM_BASE_URL=
LLM_MODEL=
```

`LLM_BASE_URL` 和 `LLM_MODEL` 留空时分别使用以下默认值：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `qwen-plus`

也可以直接设置同名系统环境变量。不要提交 `.env` 文件。

## 运行

```bash
python main.py "你好，请介绍一下你自己"
```

终端输出一段模型回复即表示 v0.1 的完整调用链工作正常。

不传入 prompt 时，程序会显示帮助和运行示例：

```bash
python main.py
```

## 文件职责

- `config.py`：读取和校验环境变量配置。
- `llm.py`：封装一次普通的 Chat Completions 模型调用。
- `main.py`：处理命令行输入、输出和用户可读错误。
- `requirements.txt`：记录最小运行依赖。
- `.env.example`：提供不含密钥的环境变量模板。
- `.gitignore`：避免提交密钥、本地虚拟环境和 Python 缓存。
