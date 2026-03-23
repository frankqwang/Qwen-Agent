"""Native Qwen-Agent WebUI with STS2 MCP tools."""

from __future__ import annotations

import argparse
import html
import json
import pprint
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qwen_agent.agents.fncall_agent as fncall_agent_module
from qwen_agent.agents import Assistant
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui import WebUI
from qwen_agent.gui.gradio_dep import gr, mgr, ms
from qwen_agent.gui.gradio_utils import covert_image_to_base64
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, USER, VIDEO
from qwen_agent.log import logger


DEFAULT_SYSTEM_MESSAGE = """你正在通过 MCP 工具游玩《杀戮尖塔2》。
请始终以工具返回的游戏状态为准，不要编造卡牌、敌人、地图、奖励或事件内容。
你可以在一轮里连续执行多个真实动作，但要谨慎，优先完成当前战斗回合、奖励领取或单个界面的合理推进。
如果工具调用失败，请用中文简短说明原因，并尝试更安全的下一步。
默认用中文回答，说明尽量简洁直接。"""

APP_BOT_CSS = (REPO_ROOT / "qwen_agent" / "gui" / "assets" / "appBot.css").read_text(encoding="utf-8")

STS2_WEBUI_CSS = APP_BOT_CSS + """
html,body{margin:0!important;padding:0!important;background:#111318!important}
body .gradio-container{max-width:none!important;width:calc(100vw - 2px)!important;margin:0!important;padding:1px!important}
body .gradio-container .main{max-width:none!important;width:100%!important;margin:0!important;padding:0!important}
.app,.wrap,.contain{max-width:none!important;width:100%!important;margin:0!important;padding:0!important}
.container{display:grid!important;grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;gap:10px!important;width:100%!important;max-width:none!important;margin:0!important;padding:0!important}
.sts2-main-col,.sts2-side-col{min-width:0!important;width:100%!important;max-width:none!important}
.sts2-chatbot,.sts2-chatbot>div{width:100%!important;min-width:0!important}
.sts2-chatbot .message-wrap,.sts2-chatbot .message-row,.sts2-chatbot .bubble-wrap{max-width:100%!important;width:100%!important}
.sts2-chatbot .message{max-width:100%!important;width:100%!important}
.sts2-chatbot .avatar-container,.sts2-chatbot .avatar-image-container{min-width:48px!important;width:48px!important}
.sts2-sidebar{gap:10px}
.sts2-agent-card{display:flex;align-items:center;gap:10px;padding:12px;border:1px solid #243041;border-radius:12px;background:#101828;color:#f8fafc}
.sts2-agent-card__avatar img{width:40px;height:40px;border-radius:999px;object-fit:cover;display:block}
.sts2-agent-card__title{font-size:16px;font-weight:700;line-height:1.2;color:#f8fafc}
.sts2-agent-card__desc{margin-top:4px;font-size:12px;line-height:1.45;color:#98a2b3}
.sts2-panel{padding:14px;border:1px solid #243041;border-radius:12px;background:#101828;color:#e5e7eb;box-sizing:border-box;width:100%!important}
.sts2-panel__title{margin-bottom:10px;font-size:14px;font-weight:700;color:#f8fafc}
.sts2-panel__summary{font-size:13px;line-height:1.5;color:#d0d5dd}
.sts2-state-lines{display:flex;flex-wrap:wrap;gap:6px 10px;margin-top:10px}
.sts2-state-chip{font-size:12px;line-height:1.35;color:#d0d5dd;background:#0b1220;border:1px solid #243041;border-radius:999px;padding:4px 10px}
.sts2-state-detail{margin-top:8px;font-size:12px;line-height:1.45;color:#d0d5dd}
.sts2-panel__footnote{margin-top:10px;font-size:12px;color:#98a2b3}
.sts2-control-row{gap:8px}
.sts2-control-row>*{min-width:0!important}
.sts2-small-btn button{min-height:34px!important;height:34px!important;padding:0 12px!important;font-size:13px!important;border-radius:9px!important}
.sts2-number-compact input{min-height:36px!important}
.sts2-autoplay-inline{padding:2px 2px 0 2px;font-size:12px;color:#98a2b3}
.sts2-autoplay-inline strong{color:#f8fafc}
.sts2-step-card{padding:8px 10px;border:1px solid #243041;border-radius:10px;background:#0b1220}
.sts2-step-card+.sts2-step-card{margin-top:10px}
.sts2-step-card__title{font-size:12px;font-weight:700;color:#f8fafc}
.sts2-step-card__line{margin-top:4px;font-size:12px;line-height:1.45;color:#d0d5dd}
.sts2-steps-scroll{max-height:325px;overflow-y:auto;padding-right:4px}
.markdown-body details{white-space:normal!important;overflow-wrap:anywhere;word-break:break-word}
.markdown-body summary{white-space:normal!important}
.markdown-body .message{overflow-wrap:anywhere;word-break:break-word}
@media (max-width: 1100px){.container{grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;gap:8px!important}}
"""


def fetch_state(game_host: str, game_port: int) -> dict[str, Any]:
    query = urllib.parse.urlencode({"format": "json"})
    req = urllib.request.Request(
        f"http://{game_host}:{game_port}/api/v1/singleplayer?{query}",
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_player(state: dict[str, Any]) -> dict[str, Any]:
    if isinstance(state.get("player"), dict):
        return state["player"]
    for key in ("battle", "map", "rewards", "rest_site", "shop", "event", "card_select", "relic_select", "treasure"):
        section = state.get(key)
        if isinstance(section, dict) and isinstance(section.get("player"), dict):
            return section["player"]
    return {}


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(part.strip() for part in parts if str(part).strip()).strip()


def summarize_state(state: dict[str, Any]) -> str:
    state_type = state.get("state_type", "unknown")
    run_status = state.get("run_status", "unknown")
    player = get_player(state)
    hp = player.get("hp", "-")
    max_hp = player.get("max_hp", "-")
    gold = player.get("gold", "-")
    floor_num = state.get("floor") or state.get("floor_num") or state.get("act_floor") or "-"
    return f"状态 {state_type}，运行 {run_status}，楼层 {floor_num}，生命 {hp}/{max_hp}，金币 {gold}"


def build_state_summary_for_prompt(state: dict[str, Any]) -> str:
    player = get_player(state)
    state_type = state.get("state_type", "unknown")
    lines = [
        f"当前状态类型：{state_type}",
        f"玩家：生命 {player.get('hp', '-')}/{player.get('max_hp', '-')}, 金币 {player.get('gold', '-')}",
    ]
    floor_num = state.get("floor") or state.get("floor_num") or state.get("act_floor")
    if floor_num not in (None, ""):
        lines.append(f"楼层：{floor_num}")
    if state_type in {"monster", "elite", "boss"}:
        battle = state.get("battle", {}) if isinstance(state.get("battle"), dict) else {}
        battle_player = battle.get("player", {}) if isinstance(battle.get("player"), dict) else {}
        lines.append(f"能量：{battle_player.get('energy', '-')}")
        enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
        if enemies:
            lines.append(
                "敌人：" + "；".join(
                    f"{enemy.get('entity_id') or enemy.get('name')}: {enemy.get('hp', '?')}/{enemy.get('max_hp', '?')}"
                    for enemy in enemies[:6]
                    if isinstance(enemy, dict)
                )
            )
    return "\n".join(lines)


def render_state_panel_html(game_host: str, game_port: int) -> str:
    try:
        state = fetch_state(game_host, game_port)
    except Exception as exc:  # noqa: BLE001
        return (
            "<div class='sts2-panel'>"
            "<div class='sts2-panel__title'>当前游戏状态</div>"
            f"<div class='sts2-panel__summary' style='color:#f97066;'>当前不可用：{html.escape(str(exc))}</div>"
            f"<div class='sts2-panel__footnote'>来源：http://{game_host}:{game_port}</div>"
            "</div>"
        )

    state_type = state.get("state_type", "unknown")
    player = get_player(state)
    floor_num = state.get("floor") or state.get("floor_num") or state.get("act_floor") or "-"
    chips = [
        f"状态 {state_type}",
        f"楼层 {floor_num}",
        f"生命 {player.get('hp', '-')}/{player.get('max_hp', '-')}",
        f"金币 {player.get('gold', '-')}",
    ]
    details: list[str] = []
    if state_type in {"monster", "elite", "boss"}:
        battle = state.get("battle", {}) if isinstance(state.get("battle"), dict) else {}
        energy = battle.get("player", {}).get("energy") if isinstance(battle.get("player"), dict) else "-"
        enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
        chips.extend([f"能量 {energy}", f"敌人 {len(enemies)}"])
        if enemies:
            details.append(
                "目标："
                + "，".join(
                    f"{enemy.get('name') or enemy.get('entity_id')} {enemy.get('hp')}/{enemy.get('max_hp')}"
                    for enemy in enemies[:2]
                    if isinstance(enemy, dict)
                )
            )
    chips_html = "".join(f"<div class='sts2-state-chip'>{html.escape(item)}</div>" for item in chips)
    details_html = "".join(f"<div class='sts2-state-detail'>{html.escape(item)}</div>" for item in details[:2])
    return (
        "<div class='sts2-panel'>"
        "<div class='sts2-panel__title'>当前游戏状态</div>"
        f"<div class='sts2-panel__summary'>{html.escape(summarize_state(state))}</div>"
        f"<div class='sts2-state-lines'>{chips_html}</div>"
        f"{details_html}"
        f"<div class='sts2-panel__footnote'>来源：http://{game_host}:{game_port}</div>"
        "</div>"
    )


class STS2WebUI(WebUI):
    def __init__(self, *args, game_host: str, game_port: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_host = game_host
        self.game_port = game_port
        self.autoplay_enabled = False
        self.auto_step_counter = 0
        self.recent_steps: list[dict[str, str]] = []

    def _message_get(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def expand_debug_sections(self, content: Any) -> Any:
        if not isinstance(content, str):
            return content
        return content.replace(
            "<details>\n  <summary>Thinking ...</summary>",
            "<details open>\n  <summary>Thinking ...</summary>",
        )

    def configure_run_limits(self, max_actions_per_round: int) -> None:
        fncall_agent_module.MAX_LLM_CALL_PER_RUN = max(6, max_actions_per_round * 3)

    def build_auto_prompt(self, max_actions_per_round: int, state_mode: str) -> str:
        prompt_lines = [
            "继续自动游玩这一局。",
            f"本轮最多允许你连续执行 {max_actions_per_round} 个真实游戏动作。",
        ]
        state = self.fetch_runtime_state()
        if state_mode == "json_summary" and state:
            prompt_lines.append("下面是基于本地 JSON 状态整理的摘要，可直接据此决策；必要时也可以再调用工具核对状态。")
            prompt_lines.append(build_state_summary_for_prompt(state))
        else:
            prompt_lines.append("请优先调用 `sts2-get_game_state` 读取最新状态，再开始行动。")
        prompt_lines.append("尽量把当前这一屏或当前回合推进得更完整一些，不要只做一个最小动作就停下。")
        prompt_lines.append("最后用中文简短说明你做了什么。")
        return "\n".join(prompt_lines)

    def render_state_panel(self) -> str:
        return render_state_panel_html(self.game_host, self.game_port)

    def render_autoplay_status(self, status: str, detail: str = "") -> str:
        text = f"自动运行：{status}"
        if detail:
            text += f" | {detail}"
        return f"<div class='sts2-autoplay-inline'><strong>{html.escape(text)}</strong></div>"

    def render_recent_steps_html(self) -> str:
        if not self.recent_steps:
            body = "<div class='sts2-panel__summary'>暂无步骤记录。</div>"
        else:
            cards = []
            for item in reversed(self.recent_steps[-50:]):
                cards.append(
                    "<div class='sts2-step-card'>"
                    f"<div class='sts2-step-card__title'>{html.escape(item['title'])}</div>"
                    f"<div class='sts2-step-card__line'>{html.escape(item['summary'])}</div>"
                    "</div>"
                )
            body = f"<div class='sts2-steps-scroll'>{''.join(cards)}</div>"
        return "<div class='sts2-panel'><div class='sts2-panel__title'>最近步骤</div>" + body + "</div>"

    def render_agent_info_html(self, agent_index: int = 0) -> str:
        agent_config = self.agent_config_list[agent_index]
        avatar = agent_config.get("avatar") or get_avatar_image(agent_config["name"])
        return (
            "<div class='sts2-agent-card'>"
            f"<div class='sts2-agent-card__avatar'><img src='{covert_image_to_base64(avatar)}' /></div>"
            "<div>"
            f"<div class='sts2-agent-card__title'>{html.escape(agent_config['name'])}</div>"
            f"<div class='sts2-agent-card__desc'>{html.escape(agent_config['description'])}</div>"
            "</div>"
            "</div>"
        )

    def fetch_runtime_state(self) -> dict[str, Any] | None:
        try:
            return fetch_state(self.game_host, self.game_port)
        except Exception:
            return None

    def state_signature(self, state: dict[str, Any] | None) -> str:
        if not state:
            return "null"
        player = get_player(state)
        battle = state.get("battle", {}) if isinstance(state.get("battle"), dict) else {}
        enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
        payload = {
            "state_type": state.get("state_type"),
            "run_status": state.get("run_status"),
            "floor": state.get("floor") or state.get("floor_num") or state.get("act_floor"),
            "hp": player.get("hp"),
            "gold": player.get("gold"),
            "energy": battle.get("player", {}).get("energy") if isinstance(battle.get("player"), dict) else None,
            "enemy_hp": [(enemy.get("entity_id"), enemy.get("hp")) for enemy in enemies if isinstance(enemy, dict)],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def stop_reason(self, state: dict[str, Any] | None) -> str | None:
        if not state:
            return None
        run_status = str(state.get("run_status") or "").lower()
        if run_status == "victory":
            return "本局已胜利"
        if run_status == "defeat":
            return "本局已失败"
        if state.get("state_type") == "menu":
            return "已回到主菜单"
        return None

    def tool_action_cn(self, name: str, arguments: str | None) -> str:
        args: dict[str, Any] = {}
        if arguments:
            try:
                loaded = json.loads(arguments)
                if isinstance(loaded, dict):
                    args = loaded
            except Exception:
                pass
        mapping = {
            "sts2-get_game_state": "读取当前状态",
            "sts2-combat_end_turn": "结束回合",
            "sts2-use_potion": "使用药水",
            "sts2-rewards_claim": "领取奖励",
            "sts2-rewards_skip_card": "跳过卡牌奖励",
            "sts2-proceed_to_map": "前往地图",
            "sts2-event_advance_dialogue": "继续事件对话",
            "sts2-deck_confirm_selection": "确认当前选择",
            "sts2-deck_cancel_selection": "取消当前选择",
            "sts2-relic_skip": "跳过遗物",
        }
        if name in mapping:
            return mapping[name]
        if name == "sts2-combat_play_card":
            text = f"打出第 {args.get('card_index', '?')} 张牌"
            if args.get("target"):
                text += f"，目标 {args.get('target')}"
            return text
        if name == "sts2-map_choose_node":
            return f"选择地图节点 {args.get('node_index', '?')}"
        if name == "sts2-rewards_pick_card":
            return f"选择奖励卡牌 {args.get('card_index', '?')}"
        if name == "sts2-rest_choose_option":
            return f"选择篝火选项 {args.get('option_index', '?')}"
        if name == "sts2-shop_purchase":
            return f"购买商店物品 {args.get('item_index', '?')}"
        if name == "sts2-event_choose_option":
            return f"选择事件选项 {args.get('option_index', '?')}"
        if name == "sts2-deck_select_card":
            return f"选择卡牌 {args.get('card_index', '?')}"
        if name == "sts2-relic_select":
            return f"选择遗物 {args.get('relic_index', '?')}"
        if name == "sts2-treasure_claim_relic":
            return f"拾取遗物 {args.get('relic_index', '?')}"
        return name.replace("sts2-", "")

    def short_state_cn(self, state: dict[str, Any] | None) -> str:
        if not state:
            return "状态未知"
        state_type = state.get("state_type", "unknown")
        player = get_player(state)
        hp = player.get("hp", "-")
        max_hp = player.get("max_hp", "-")
        gold = player.get("gold", "-")
        if state_type in {"monster", "elite", "boss"}:
            battle = state.get("battle", {}) if isinstance(state.get("battle"), dict) else {}
            energy = battle.get("player", {}).get("energy") if isinstance(battle.get("player"), dict) else "-"
            enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
            return f"{state_type}，生命 {hp}/{max_hp}，能量 {energy}，敌人 {len(enemies)}，金币 {gold}"
        return f"{state_type}，生命 {hp}/{max_hp}，金币 {gold}"

    def record_recent_step(
        self,
        title: str,
        pre_state: dict[str, Any] | None,
        post_state: dict[str, Any] | None,
        turn_messages: list[Any],
    ) -> None:
        actions: list[str] = []
        fallback_text = ""
        for message in turn_messages:
            role = self._message_get(message, ROLE)
            if role == "assistant":
                function_call = self._message_get(message, "function_call")
                if function_call:
                    name = self._message_get(function_call, "name")
                    arguments = self._message_get(function_call, "arguments")
                    if name:
                        actions.append(self.tool_action_cn(str(name), arguments))
                if not fallback_text:
                    fallback_text = extract_text_content(self._message_get(message, CONTENT))
            elif role == "function" and not actions:
                name = self._message_get(message, NAME)
                if name:
                    actions.append(self.tool_action_cn(str(name), None))

        unique_actions = [item for item in dict.fromkeys(actions) if item]
        if unique_actions and unique_actions[0] == "读取当前状态" and len(unique_actions) > 1:
            action_text = f"读取状态后，{'；'.join(unique_actions[1:3])}"
        elif unique_actions:
            action_text = "；".join(unique_actions[:3])
        else:
            fallback_text = fallback_text.replace("\n", " ").strip()
            action_text = fallback_text[:80] if fallback_text else "未执行明确动作"

        post_text = self.short_state_cn(post_state)
        pre_text = self.short_state_cn(pre_state)
        summary = f"{action_text}。"
        summary += "状态基本未变化。" if pre_text == post_text else f"当前{post_text}。"
        self.recent_steps.append({"title": title, "summary": summary[:220]})
        self.recent_steps = self.recent_steps[-50:]

    def append_auto_user_turn(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        title: str,
        max_actions_per_round: int,
        state_mode: str,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        chatbot = list(chatbot or [])
        history = list(history or [])
        prompt = self.build_auto_prompt(max_actions_per_round=max_actions_per_round, state_mode=state_mode)
        history.append({ROLE: USER, CONTENT: [{"text": prompt}], NAME: self.user_config[NAME]})
        chatbot.append([f"[{title}]", None])
        return chatbot, history

    def _derive_turn_title(self, history: list[dict[str, Any]]) -> str:
        for message in reversed(history):
            if self._message_get(message, ROLE) != USER:
                continue
            text = extract_text_content(self._message_get(message, CONTENT))
            if not text:
                continue
            if text.startswith("[自动步骤"):
                return text.strip("[]")
            compact = text.replace("\n", " ").strip()
            if len(compact) > 18:
                compact = compact[:18] + "..."
            return compact or "手动步骤"
        return "手动步骤"

    def _run_agent_once(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        agent_selector: int = 0,
        max_actions_per_round: int = 3,
        expand_debug: bool = True,
    ) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
        chatbot = list(chatbot or [])
        history = list(history or [])
        self.configure_run_limits(max_actions_per_round)
        agent_count = len(self.agent_list)
        if not chatbot:
            chatbot = [[None, [None for _ in range(agent_count)]]]
        if chatbot[-1][1] is None:
            chatbot[-1][1] = [None for _ in range(agent_count)]

        if self.verbose:
            logger.info("agent_run input:\n" + pprint.pformat(history, indent=2))

        num_input_bubbles = len(chatbot) - 1
        num_output_bubbles = 1
        responses: list[Any] = []
        agent_runner = self.agent_list[agent_selector]
        if self.agent_hub:
            agent_runner = self.agent_hub

        for responses in agent_runner.run(history, **self.run_kwargs):
            if not responses:
                continue
            last_content = self._message_get(responses[-1], CONTENT)
            if last_content == PENDING_USER_INPUT:
                break

        final_messages = [res for res in responses if self._message_get(res, CONTENT) != PENDING_USER_INPUT]
        if final_messages:
            display_responses = convert_fncall_to_text(final_messages)
            if expand_debug:
                for rsp in display_responses:
                    if isinstance(rsp, dict) and CONTENT in rsp:
                        rsp[CONTENT] = self.expand_debug_sections(rsp[CONTENT])
            while len(display_responses) > num_output_bubbles:
                chatbot.append([None, [None for _ in range(agent_count)]])
                num_output_bubbles += 1
            for index in range(num_output_bubbles):
                if chatbot[num_input_bubbles + index][1] is None:
                    chatbot[num_input_bubbles + index][1] = [None for _ in range(agent_count)]
            for index, rsp in enumerate(display_responses):
                agent_index = 0
                if isinstance(rsp, dict):
                    agent_index = self._get_agent_index_by_name(rsp.get(NAME))
                    chatbot[num_input_bubbles + index][1][agent_index] = rsp.get(CONTENT)
            history.extend(final_messages)

        return chatbot, history, final_messages

    def _complete_turn(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        title: str,
        agent_selector: int = 0,
        max_actions_per_round: int = 3,
        expand_debug: bool = True,
    ) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        pre_state = self.fetch_runtime_state()
        chatbot, history, turn_messages = self._run_agent_once(
            chatbot,
            history,
            agent_selector=agent_selector,
            max_actions_per_round=max_actions_per_round,
            expand_debug=expand_debug,
        )
        post_state = self.fetch_runtime_state()
        self.record_recent_step(title, pre_state, post_state, turn_messages)
        return chatbot, history, pre_state, post_state

    def single_step(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        max_actions_per_round: int,
        state_mode: str,
        expand_debug: bool,
        agent_selector: int = 0,
    ):
        self.auto_step_counter += 1
        title = f"自动步骤 {self.auto_step_counter}"
        chatbot, history = self.append_auto_user_turn(chatbot, history, title, int(max_actions_per_round), state_mode)
        chatbot, history, _, post_state = self._complete_turn(
            chatbot,
            history,
            title,
            agent_selector=agent_selector,
            max_actions_per_round=int(max_actions_per_round),
            expand_debug=expand_debug,
        )
        detail = self.stop_reason(post_state) or "单步完成"
        yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("空闲", detail)

    def start_autoplay(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        max_steps: int | float | None,
        max_stuck: int | float | None,
        max_actions_per_round: int | float | None,
        state_mode: str,
        expand_debug: bool,
        agent_selector: int = 0,
    ):
        limit = int(max_steps or 0)
        stuck_limit = max(1, int(max_stuck or 6))
        max_actions = max(1, int(max_actions_per_round or 3))
        unchanged_rounds = 0
        self.autoplay_enabled = True
        yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("运行中", "准备开始")

        while self.autoplay_enabled:
            current_state = self.fetch_runtime_state()
            reason = self.stop_reason(current_state)
            if reason:
                self.autoplay_enabled = False
                yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已停止", reason)
                return
            if limit > 0 and self.auto_step_counter >= limit:
                self.autoplay_enabled = False
                yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已停止", f"达到最大步数 {limit}")
                return

            self.auto_step_counter += 1
            title = f"自动步骤 {self.auto_step_counter}"
            chatbot, history = self.append_auto_user_turn(chatbot, history, title, max_actions, state_mode)
            yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("运行中", f"{title} 进行中")

            chatbot, history, pre_state, post_state = self._complete_turn(
                chatbot,
                history,
                title,
                agent_selector=agent_selector,
                max_actions_per_round=max_actions,
                expand_debug=expand_debug,
            )
            unchanged_rounds = unchanged_rounds + 1 if self.state_signature(pre_state) == self.state_signature(post_state) else 0
            reason = self.stop_reason(post_state)
            if reason:
                self.autoplay_enabled = False
                yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已停止", reason)
                return
            if unchanged_rounds >= stuck_limit:
                self.autoplay_enabled = False
                yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已停止", f"连续 {unchanged_rounds} 轮状态未变化，已暂停")
                return

            yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("运行中", f"已执行 {self.auto_step_counter} 步")
            time.sleep(0.15)

        yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已暂停", f"已执行 {self.auto_step_counter} 步")

    def pause_autoplay(self):
        self.autoplay_enabled = False
        return self.render_autoplay_status("已暂停", f"已执行 {self.auto_step_counter} 步")

    def agent_run_with_state(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        max_actions_per_round: int,
        state_mode: str,
        expand_debug: bool,
        agent_selector: int = 0,
    ):
        title = self._derive_turn_title(history)
        if state_mode == "json_summary":
            state = self.fetch_runtime_state()
            if state:
                history = list(history or [])
                history.append({
                    ROLE: USER,
                    CONTENT: [{"text": "补充给你的结构化状态摘要如下，可结合工具继续决策：\n" + build_state_summary_for_prompt(state)}],
                    NAME: self.user_config[NAME],
                })
        chatbot, history, _, _ = self._complete_turn(
            chatbot,
            history,
            title,
            agent_selector=agent_selector,
            max_actions_per_round=int(max_actions_per_round),
            expand_debug=expand_debug,
        )
        yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("空闲")

    def run(
        self,
        messages: list[dict[str, Any]] | None = None,
        share: bool = False,
        server_name: str | None = None,
        server_port: int | None = None,
        concurrency_limit: int = 10,
        **kwargs,
    ):
        self.run_kwargs = kwargs
        custom_theme = gr.themes.Default(primary_hue=gr.themes.utils.colors.blue, radius_size=gr.themes.utils.sizes.radius_none)

        with gr.Blocks(css=STS2_WEBUI_CSS, theme=custom_theme) as demo:
            history = gr.State(list(messages or []))
            agent_selector = gr.State(0)
            with ms.Application():
                with gr.Row(elem_classes="container"):
                    with gr.Column(scale=1, elem_classes="sts2-main-col"):
                        chatbot = mgr.Chatbot(
                            value=convert_history_to_chatbot(messages=messages),
                            avatar_images=[self.user_config, self.agent_config_list],
                            height=850,
                            avatar_image_width=56,
                            flushing=False,
                            show_copy_button=True,
                            elem_classes=["sts2-chatbot"],
                        )
                        input_box = mgr.MultimodalInput(placeholder=self.input_placeholder)
                        audio_input = gr.Audio(sources=["microphone"], type="filepath")

                    with gr.Column(scale=1, elem_classes=["sts2-sidebar", "sts2-side-col"]):
                        gr.HTML(self.render_agent_info_html())
                        state_panel = gr.HTML(self.render_state_panel())
                        recent_steps = gr.HTML(self.render_recent_steps_html())

                        with gr.Row(elem_classes=["sts2-control-row"]):
                            single_step_btn = gr.Button("单步", elem_classes=["sts2-small-btn"])
                            auto_run_btn = gr.Button("自动", variant="primary", elem_classes=["sts2-small-btn"])
                            pause_btn = gr.Button("暂停", variant="stop", elem_classes=["sts2-small-btn"])
                            refresh_btn = gr.Button("刷新", elem_classes=["sts2-small-btn"])

                        with gr.Row(elem_classes=["sts2-control-row"]):
                            state_mode = gr.Dropdown(label="状态模式", choices=[("Markdown 原始", "markdown"), ("JSON 摘要", "json_summary")], value="markdown")
                            max_actions = gr.Number(label="每轮动作数", value=3, minimum=1, precision=0, elem_classes=["sts2-number-compact"])
                            expand_debug = gr.Checkbox(label="展开思考", value=True)

                        with gr.Row(elem_classes=["sts2-control-row"]):
                            max_steps = gr.Number(label="最大步数（0=无限）", value=0, minimum=0, precision=0, elem_classes=["sts2-number-compact"])
                            max_stuck = gr.Number(label="卡住轮数", value=6, minimum=1, precision=0, elem_classes=["sts2-number-compact"])

                        autoplay_status = gr.HTML(self.render_autoplay_status("空闲"))
                        if self.prompt_suggestions:
                            gr.Examples(label="建议提示词", examples=self.prompt_suggestions, inputs=[input_box])

                input_promise = input_box.submit(
                    fn=self.add_text,
                    inputs=[input_box, audio_input, chatbot, history],
                    outputs=[input_box, audio_input, chatbot, history],
                    queue=False,
                )
                input_promise = input_promise.then(
                    self.agent_run_with_state,
                    [chatbot, history, max_actions, state_mode, expand_debug, agent_selector],
                    [chatbot, history, state_panel, recent_steps, autoplay_status],
                )
                input_promise.then(self.flushed, None, [input_box])

                single_step_btn.click(
                    fn=self.single_step,
                    inputs=[chatbot, history, max_actions, state_mode, expand_debug, agent_selector],
                    outputs=[chatbot, history, state_panel, recent_steps, autoplay_status],
                )
                auto_run_btn.click(
                    fn=self.start_autoplay,
                    inputs=[chatbot, history, max_steps, max_stuck, max_actions, state_mode, expand_debug, agent_selector],
                    outputs=[chatbot, history, state_panel, recent_steps, autoplay_status],
                )
                pause_btn.click(fn=self.pause_autoplay, inputs=None, outputs=[autoplay_status], queue=False)
                refresh_btn.click(
                    fn=lambda: (
                        self.render_state_panel(),
                        self.render_recent_steps_html(),
                        self.render_autoplay_status("运行中" if self.autoplay_enabled else "空闲", f"已执行 {self.auto_step_counter} 步"),
                    ),
                    inputs=None,
                    outputs=[state_panel, recent_steps, autoplay_status],
                    queue=False,
                )

            demo.load(None)

        demo.queue(default_concurrency_limit=concurrency_limit).launch(share=share, server_name=server_name, server_port=server_port)


def build_llm_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
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


def build_mcp_tools(args: argparse.Namespace) -> list[dict[str, Any]]:
    server_script = Path(args.mcp_server_script).resolve()
    server_dir = Path(args.mcp_server_dir).resolve()
    return [{
        "mcpServers": {
            "sts2": {
                "command": args.mcp_python,
                "args": [str(server_script), "--host", args.game_host, "--port", str(args.game_port)],
                "cwd": str(server_dir),
            }
        }
    }]


def build_agent(args: argparse.Namespace) -> Assistant:
    return Assistant(
        llm=build_llm_cfg(args),
        name="STS2 MCP Agent",
        description="原生 Qwen-Agent 界面，连接 Slay the Spire 2 MCP 工具。",
        system_message=args.system_message or DEFAULT_SYSTEM_MESSAGE,
        function_list=build_mcp_tools(args),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the native Qwen-Agent WebUI for Slay the Spire 2.")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--api-base", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-input-tokens", type=int, default=16000)
    parser.add_argument("--fncall-prompt-type", default="nous")
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM_MESSAGE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--game-host", default="localhost")
    parser.add_argument("--game-port", type=int, default=15526)
    parser.add_argument("--mcp-python", default=sys.executable)
    parser.add_argument("--mcp-server-dir", default=r"C:\dev\STS2MCP\mcp")
    parser.add_argument("--mcp-server-script", default=r"C:\dev\STS2MCP\mcp\server.py")
    args = parser.parse_args()

    bot = build_agent(args)
    chatbot_config = {
        "input.placeholder": "让 agent 查看 STS2 状态，或者直接执行下一步。",
        "prompt.suggestions": [
            "先读取当前游戏状态，然后给出下一步最合适的动作。",
            "请继续当前这局游戏，先读状态再行动。",
            "如果当前是战斗，优先考虑最稳的出牌顺序。",
        ],
    }

    ui = STS2WebUI(bot, chatbot_config=chatbot_config, game_host=args.game_host, game_port=args.game_port)
    print(f"Game API: http://{args.game_host}:{args.game_port}")
    print(f"Qwen-Agent UI: http://{args.host}:{args.port}")
    ui.run(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
