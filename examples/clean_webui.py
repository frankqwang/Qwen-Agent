"""Launch a clean Qwen-Agent WebUI backed by a local OpenAI-compatible model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI


def build_agent(args: argparse.Namespace) -> Assistant:
    llm_cfg = {
        "model": args.model,
        "model_type": "oai",
        "model_server": args.api_base,
        "api_key": args.api_key,
        "generate_cfg": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_input_tokens": args.max_input_tokens,
            "fncall_prompt_type": args.fncall_prompt_type,
        },
    }
    return Assistant(
        llm=llm_cfg,
        name=args.name,
        description=args.description,
        system_message=args.system_message or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a clean Qwen-Agent WebUI with a local OpenAI-compatible model.")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--api-base", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-input-tokens", type=int, default=16000)
    parser.add_argument("--fncall-prompt-type", default="nous")
    parser.add_argument("--name", default="Qwen")
    parser.add_argument("--description", default="Clean Qwen-Agent WebUI")
    parser.add_argument("--system-message", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    bot = build_agent(args)
    print("Starting clean Qwen-Agent WebUI.")
    print(f"Model endpoint: {args.api_base}")
    print(f"WebUI: http://{args.host}:{args.port}")

    WebUI(bot).run(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
