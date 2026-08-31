"""Command-line entry point for the v0.3 coding agent."""

import argparse

from agent import AgentResult, AgentStatus, CodingAgent
from config import ConfigurationError
from llm import LLMClient, LLMError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="让 Coding Agent 在本地 workspace 中处理一项任务。"
    )
    parser.add_argument("task", nargs="?", help="希望 Agent 完成的任务")
    return parser


def print_result(result: AgentResult) -> None:
    print(f"\n[Status] {result.status.value}")
    print("[Final]")
    print(result.final_answer)

    if result.verification_evidence:
        print("\n[Verification]")
        for evidence in result.verification_evidence:
            state = "passed" if evidence.success else "failed"
            print(f"- Step {evidence.step}: {evidence.command}: {state}")

    if result.error and result.status is not AgentStatus.COMPLETED:
        print(f"\n[Reason] {result.error}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.task or not args.task.strip():
        parser.print_help()
        print('\n示例：python main.py "Inspect the workspace and explain this project."')
        return 0

    try:
        client = LLMClient()
        agent = CodingAgent(client)
    except (ConfigurationError, LLMError) as exc:
        print("[Status] FATAL_ERROR")
        print(f"[Reason] {exc}")
        return 1

    result = agent.run(args.task)
    print_result(result)
    if result.status is AgentStatus.COMPLETED:
        return 0
    if result.status is AgentStatus.FATAL_ERROR:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
