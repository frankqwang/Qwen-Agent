"""Autoplay Slay the Spire 2 with a local OpenAI-compatible model and MCP tools.

This example is designed for local setups such as LM Studio + Qwen models.
It uses:
1. Qwen-Agent Assistant for tool use.
2. The STS2 MCP server as the tool surface.
3. A lightweight outer control loop to keep the agent on short, stable decision cycles.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import copy
from collections import deque
from pathlib import Path
from typing import Deque

# Allow `python examples\sts2_autoplay_agent.py` from the repo root without
# requiring an editable install first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, Message


SYSTEM_PROMPT = """You are a cautious gameplay agent whose only goal is to win the
current Slay the Spire 2 run.

Rules:
1. At the start of every decision cycle, call `sts2-get_game_state` first and
   prefer `format="json"`.
2. Execute at most one game-state-changing action per cycle, then stop.
3. If needed, you may call `sts2-get_game_state` again after the action to
   confirm the result.
4. Never invent game state, card effects, enemy intents, map nodes, or reward
   contents. Use tool outputs only.
5. If an action needs a target, always use the exact `target` or `entity_id`
   returned by the tools.
6. Do not call combat tools when the current screen is not combat.
7. If there is no safe action, read the state again and explain the blocker
   instead of guessing.
8. Prefer stable play: preserve HP, avoid pointless greed, and choose lines
   that improve run consistency.
9. For card rewards, events, shop choices, rest sites, and route choices,
   optimize for current-run win rate, not flashy combos.
10. Your final reply must be short and contain exactly:
state: ...
action: ...
why: ...

You are an execution agent, not a commentator. Advance the run by one step per
cycle."""


class NonStreamingAssistant(Assistant):

    def _run(self, messages: list[Message], lang: str = "en", **kwargs):
        messages = copy.deepcopy(messages)
        num_llm_calls_available = 10
        response: list[Message] = []

        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            extra_generate_cfg = {"lang": lang}
            if kwargs.get("seed") is not None:
                extra_generate_cfg["seed"] = kwargs["seed"]

            output = self._call_llm(
                messages=messages,
                functions=[func.function for func in self.function_map.values()],
                stream=False,
                extra_generate_cfg=extra_generate_cfg,
            )

            if not output:
                break

            response.extend(output)
            messages.extend(output)
            yield response

            used_any_tool = False
            for out in output:
                use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                if use_tool:
                    tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                    fn_msg = Message(
                        role=FUNCTION,
                        name=tool_name,
                        content=tool_result,
                        extra={"function_id": out.extra.get("function_id", "1")},
                    )
                    messages.append(fn_msg)
                    response.append(fn_msg)
                    yield response
                    used_any_tool = True

            if not used_any_tool:
                break

        yield response


def build_llm_cfg(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "model_type": "oai",
        "model_server": args.api_base,
        "api_key": args.api_key,
        "generate_cfg": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_input_tokens": args.max_input_tokens,
            "fncall_prompt_type": "nous",
        },
    }


def build_mcp_tools(args: argparse.Namespace) -> list[dict]:
    return [{
        "mcpServers": {
            "sts2": {
                "command": args.uv_command,
                "args": [
                    "run",
                    "--directory",
                    args.sts2_mcp_dir,
                    "python",
                    "server.py",
                    "--host",
                    args.game_host,
                    "--port",
                    str(args.game_port),
                ],
            }
        }
    }]


def init_agent(args: argparse.Namespace) -> Assistant:
    return NonStreamingAssistant(
        llm=build_llm_cfg(args),
        function_list=build_mcp_tools(args),
        name="STS2 Autoplay Agent",
        description="Play Slay the Spire 2 through the STS2 MCP server.",
        system_message=SYSTEM_PROMPT,
    )


def fetch_state(game_host: str, game_port: int) -> dict:
    query = urllib.parse.urlencode({"format": "json"})
    url = f"http://{game_host}:{game_port}/api/v1/singleplayer?{query}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def state_signature(state: dict) -> str:
    state_type = state.get("state_type", "unknown")
    run_status = state.get("run_status")
    run = state.get("run", {}) if isinstance(state.get("run"), dict) else {}

    player = {}
    if isinstance(state.get("player"), dict):
        player = state.get("player", {})
    elif isinstance(state.get(state_type), dict) and isinstance(state[state_type].get("player"), dict):
        player = state[state_type]["player"]
    elif isinstance(state.get("map"), dict) and isinstance(state["map"].get("player"), dict):
        player = state["map"]["player"]

    enemies = state.get("enemies", []) if isinstance(state.get("enemies"), list) else []
    hp = player.get("hp")
    max_hp = player.get("max_hp")
    gold = player.get("gold")
    floor_num = state.get("floor") or state.get("floor_num") or state.get("act_floor") or run.get("floor")
    enemy_sig = [(e.get("entity_id"), e.get("hp"), e.get("block")) for e in enemies if isinstance(e, dict)]
    return json.dumps(
        {
            "state_type": state_type,
            "run_status": run_status,
            "hp": hp,
            "max_hp": max_hp,
            "gold": gold,
            "floor": floor_num,
            "enemies": enemy_sig,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def summarize_state(state: dict) -> str:
    state_type = state.get("state_type", "unknown")
    run_status = state.get("run_status", "unknown")
    run = state.get("run", {}) if isinstance(state.get("run"), dict) else {}

    player = {}
    if isinstance(state.get("player"), dict):
        player = state.get("player", {})
    elif isinstance(state.get(state_type), dict) and isinstance(state[state_type].get("player"), dict):
        player = state[state_type]["player"]
    elif isinstance(state.get("map"), dict) and isinstance(state["map"].get("player"), dict):
        player = state["map"]["player"]

    hp = player.get("hp")
    max_hp = player.get("max_hp")
    gold = player.get("gold")
    floor_num = state.get("floor") or state.get("floor_num") or state.get("act_floor") or run.get("floor")
    return f"state_type={state_type}, run_status={run_status}, floor={floor_num}, hp={hp}/{max_hp}, gold={gold}"


def build_user_prompt(step: int, recent_notes: Deque[str]) -> str:
    history_block = "\n".join(f"- {note}" for note in recent_notes) if recent_notes else "- none yet"
    return f"""Goal: keep playing the current Slay the Spire 2 run and maximize the
chance to win the run.
This is decision cycle {step}.
Recent cycle summaries:
{history_block}

Requirements for this cycle:
1. Read the latest game state first.
2. Determine the current `state_type`.
3. Execute at most one real action.
4. Reply in this exact format:
state: ...
action: ...
why: ...
"""


def extract_last_assistant_text(responses: list[Message]) -> str:
    for msg in reversed(responses):
        if msg.role == ASSISTANT and isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Autoplay STS2 with Qwen-Agent and a local model.")
    parser.add_argument("--model", default="qwen/qwen3.5-4b-thinking-2507")
    parser.add_argument("--api-base", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-input-tokens", type=int, default=16000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-stuck-rounds", type=int, default=8)
    parser.add_argument("--game-host", default="localhost")
    parser.add_argument("--game-port", type=int, default=15526)
    parser.add_argument("--sts2-mcp-dir", default=r"C:\dev\STS2MCP\mcp")
    parser.add_argument("--uv-command", default="uv")
    args = parser.parse_args()

    agent = init_agent(args)
    recent_notes: Deque[str] = deque(maxlen=8)
    last_signature = None
    stuck_rounds = 0

    print("Starting STS2 autoplay loop.")
    print(f"Model endpoint: {args.api_base}")
    print(f"MCP server dir: {args.sts2_mcp_dir}")

    for step in range(1, args.max_steps + 1):
        try:
            state = fetch_state(args.game_host, args.game_port)
        except urllib.error.URLError as exc:
            print(f"[step {step}] Cannot reach game HTTP API: {exc}")
            return 1
        except json.JSONDecodeError as exc:
            print(f"[step {step}] Failed to parse game state JSON: {exc}")
            return 1

        state_type = state.get("state_type", "unknown")
        run_status = state.get("run_status")
        signature = state_signature(state)
        state_line = summarize_state(state)
        print(f"[step {step}] pre_state: {state_line}")

        if run_status == "victory":
            print("Run reported victory.")
            return 0

        if run_status == "defeat":
            print("Run reported defeat.")
            return 0

        if state_type == "menu":
            if step == 1:
                print("No run is in progress. Start or load a run first, then rerun this script.")
                return 1
            print("Run appears to have ended because the game returned to menu.")
            return 0

        if signature == last_signature:
            stuck_rounds += 1
        else:
            stuck_rounds = 0
        last_signature = signature

        if stuck_rounds >= args.max_stuck_rounds:
            print(f"State has not changed for {stuck_rounds} rounds. Stopping to avoid looping forever.")
            return 2

        user_prompt = build_user_prompt(step=step, recent_notes=recent_notes)
        responses = agent.run_nonstream(messages=[{"role": "user", "content": user_prompt}], lang="en")
        assistant_text = extract_last_assistant_text(responses) or "(no assistant summary)"
        print(f"[step {step}] {assistant_text}")
        recent_notes.append(f"step {step}: {assistant_text}")

        time.sleep(args.sleep_seconds)

    print(f"Reached max steps ({args.max_steps}). Stop here and inspect the run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
