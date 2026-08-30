"""Command-line entry point for the v0.1 LLM connectivity check."""

import argparse

from config import ConfigurationError
from llm import LLMClient, LLMError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="向通义千问发送一条消息并输出模型回复。"
    )
    parser.add_argument("prompt", nargs="?", help="要发送给模型的文本")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.prompt or not args.prompt.strip():
        parser.print_help()
        print('\n示例：python main.py "你好，请介绍一下你自己"')
        return 0

    try:
        client = LLMClient()
        response = client.chat([{"role": "user", "content": args.prompt}])
    except (ConfigurationError, LLMError) as exc:
        print(f"错误：{exc}")
        return 1

    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
