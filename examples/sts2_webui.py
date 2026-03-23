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

from qwen_agent.agents import Assistant
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui import WebUI
from qwen_agent.gui.gradio_dep import gr, mgr, ms
from qwen_agent.gui.gradio_utils import covert_image_to_base64
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, USER, VIDEO
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback


DEFAULT_SYSTEM_MESSAGE = """你正在通过 MCP 工具游玩《杀戮尖塔2》。请始终以工具返回的游戏状态为准。
行动前先读取当前状态，不要编造卡牌、敌人、地图、奖励或事件内容。
如果工具调用失败，请用中文简短说明原因，并尝试更安全的下一步。
每一轮最多执行一个真实游戏动作；只读状态不算真实动作。
默认用中文回答，说明尽量简洁直接。"""

AUTO_STEP_PROMPT = "继续自动游玩这一局。先读取当前状态，选择当前最好的单步动作并执行，然后用中文简短说明你做了什么。"

APP_BOT_CSS = (REPO_ROOT / "qwen_agent" / "gui" / "assets" / "appBot.css").read_text(encoding="utf-8")

STS2_WEBUI_CSS = APP_BOT_CSS + """
.sts2-sidebar{gap:12px}
.sts2-agent-card{display:flex;align-items:center;gap:12px;padding:12px 14px;border:1px solid #243041;border-radius:12px;background:#101828;color:#f8fafc}
.sts2-agent-card__avatar img{width:44px;height:44px;border-radius:999px;object-fit:cover;display:block}
.sts2-agent-card__title{font-size:16px;font-weight:700;line-height:1.2;color:#f8fafc}
.sts2-agent-card__desc{margin-top:4px;font-size:12px;line-height:1.45;color:#98a2b3}
.sts2-panel{padding:14px;border:1px solid #243041;border-radius:12px;background:#101828;color:#e5e7eb}
.sts2-panel__title{margin-bottom:10px;font-size:14px;font-weight:700;color:#f8fafc}
.sts2-panel__summary{font-size:13px;line-height:1.55;color:#d0d5dd;margin-bottom:12px}
.sts2-panel__grid{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;font-size:13px}
.sts2-panel__label{color:#98a2b3}
.sts2-panel__value{color:#f8fafc}
.sts2-panel__detail{margin-top:8px;font-size:13px;line-height:1.45;color:#d0d5dd}
.sts2-panel__footnote{margin-top:12px;font-size:12px;color:#98a2b3}
.sts2-state-panel{min-height:320px}
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
    extra = ""
    if state_type in {"monster", "elite", "boss"}:
        battle = state.get("battle", {}) if isinstance(state.get("battle"), dict) else {}
        energy = battle.get("player", {}).get("energy") if isinstance(battle.get("player"), dict) else "-"
        enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
        hand = battle.get("hand", []) if isinstance(battle.get("hand"), list) else []
        extra = f"，能量 {energy}，敌人 {len(enemies)}，手牌 {len(hand)}"
    elif state_type == "map":
        options = state.get("map", {}).get("next_options", []) if isinstance(state.get("map"), dict) else []
        extra = f"，可选路线 {len(options)}"
    elif state_type == "event":
        event_state = state.get("event", {}) if isinstance(state.get("event"), dict) else {}
        extra = f"，事件 {event_state.get('event_name') or '-'}，对话中 {event_state.get('in_dialogue')}"
    elif state_type == "card_select":
        section = state.get("card_select", {}) if isinstance(state.get("card_select"), dict) else {}
        extra = f"，选牌 {section.get('screen_type') or '-'}，卡牌 {len(section.get('cards', []))}"
    return f"状态 {state_type}，运行 {run_status}，楼层 {floor_num}，生命 {hp}/{max_hp}，金币 {gold}{extra}"


def render_state_panel_html(game_host: str, game_port: int) -> str:
    try:
        state = fetch_state(game_host, game_port)
    except Exception as exc:  # noqa: BLE001
        return (
            "<div class='sts2-panel sts2-state-panel'>"
            "<div class='sts2-panel__title'>当前游戏状态</div>"
            f"<div class='sts2-panel__summary' style='color:#f97066;'>当前不可用：{html.escape(str(exc))}</div>"
            f"<div class='sts2-panel__footnote'>来源：http://{game_host}:{game_port}</div>"
            "</div>"
        )

    state_type = state.get("state_type", "unknown")
    player = get_player(state)
    floor_num = state.get("floor") or state.get("floor_num") or state.get("act_floor") or "-"
    hp = player.get("hp", "-")
    max_hp = player.get("max_hp", "-")
    gold = player.get("gold", "-")
    details: list[str] = []

    if state_type in {"monster", "elite", "boss"}:
        battle = state.get("battle", {}) if isinstance(state.get("battle"), dict) else {}
        energy = battle.get("player", {}).get("energy") if isinstance(battle.get("player"), dict) else "-"
        enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
        details.append(f"能量：{energy}")
        details.append(f"敌人：{len(enemies)}")
        if enemies:
            preview = ", ".join(
                f"{enemy.get('name') or enemy.get('entity_id')} {enemy.get('hp')}/{enemy.get('max_hp')}"
                for enemy in enemies[:2]
                if isinstance(enemy, dict)
            )
            if preview:
                details.append(f"目标：{preview}")
    elif state_type == "map":
        options = state.get("map", {}).get("next_options", []) if isinstance(state.get("map"), dict) else []
        option_types = [item.get("type") for item in options if isinstance(item, dict) and item.get("type")]
        if option_types:
            details.append("下一步可选：" + ", ".join(option_types[:5]))
    elif state_type == "event":
        event_state = state.get("event", {}) if isinstance(state.get("event"), dict) else {}
        details.append(f"事件：{event_state.get('event_name') or '-'}")
        details.append(f"对话中：{event_state.get('in_dialogue')}")
    elif state_type == "card_select":
        section = state.get("card_select", {}) if isinstance(state.get("card_select"), dict) else {}
        details.append(f"界面：{section.get('screen_type') or '-'}")
        details.append(f"卡牌数：{len(section.get('cards', []))}")

    return (
        "<div class='sts2-panel sts2-state-panel'>"
        "<div class='sts2-panel__title'>当前游戏状态</div>"
        f"<div class='sts2-panel__summary'>{html.escape(summarize_state(state))}</div>"
        "<div class='sts2-panel__grid'>"
        f"<div class='sts2-panel__label'>状态</div><div class='sts2-panel__value'>{html.escape(str(state_type))}</div>"
        f"<div class='sts2-panel__label'>楼层</div><div class='sts2-panel__value'>{html.escape(str(floor_num))}</div>"
        f"<div class='sts2-panel__label'>生命</div><div class='sts2-panel__value'>{html.escape(str(hp))}/{html.escape(str(max_hp))}</div>"
        f"<div class='sts2-panel__label'>金币</div><div class='sts2-panel__value'>{html.escape(str(gold))}</div>"
        "</div>"
        + "".join(f"<div class='sts2-panel__detail'>{html.escape(item)}</div>" for item in details[:4])
        + f"<div class='sts2-panel__footnote'>来源：http://{game_host}:{game_port}</div>"
        + "</div>"
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
            return f"战斗中，生命 {hp}/{max_hp}，能量 {energy}，敌人 {len(enemies)}，金币 {gold}"
        if state_type == "map":
            return f"地图选择，生命 {hp}/{max_hp}，金币 {gold}"
        if state_type == "event":
            event_state = state.get("event", {}) if isinstance(state.get("event"), dict) else {}
            return f"事件 {event_state.get('event_name') or '-'}，生命 {hp}/{max_hp}"
        if state_type == "card_select":
            section = state.get("card_select", {}) if isinstance(state.get("card_select"), dict) else {}
            return f"选牌界面（{section.get('screen_type') or '-'}），可选 {len(section.get('cards', []))} 张"
        if state_type == "shop":
            return f"商店，金币 {gold}"
        if state_type == "rest_site":
            return f"篝火，生命 {hp}/{max_hp}"
        if state_type == "menu":
            return "主菜单"
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

        unique_actions = list(dict.fromkeys(action for action in actions if action))
        if unique_actions and unique_actions[0] == "读取当前状态" and len(unique_actions) > 1:
            action_text = f"读取状态后，{unique_actions[1]}"
        elif unique_actions:
            action_text = "；".join(unique_actions[:2])
        else:
            fallback_text = fallback_text.replace("\n", " ").strip()
            action_text = fallback_text[:80] if fallback_text else "未执行明确动作"

        post_text = self.short_state_cn(post_state)
        pre_text = self.short_state_cn(pre_state)
        summary = f"{action_text}。"
        if pre_text == post_text:
            summary += "状态基本未变化。"
        else:
            summary += f"当前{post_text}。"
        self.recent_steps.append({"title": title, "summary": summary[:220]})
        self.recent_steps = self.recent_steps[-50:]

    def append_auto_user_turn(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        title: str,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        chatbot = list(chatbot or [])
        history = list(history or [])
        history.append({ROLE: USER, CONTENT: [{"text": AUTO_STEP_PROMPT}], NAME: self.user_config[NAME]})
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
    ) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
        chatbot = list(chatbot or [])
        history = list(history or [])
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
                logger.info("Interrupted. Waiting for user input!")
                break

        final_messages = [res for res in responses if self._message_get(res, CONTENT) != PENDING_USER_INPUT]
        if final_messages:
            display_responses = convert_fncall_to_text(final_messages)
            while len(display_responses) > num_output_bubbles:
                chatbot.append([None, [None for _ in range(agent_count)]])
                num_output_bubbles += 1
            for index in range(num_output_bubbles):
                if chatbot[num_input_bubbles + index][1] is None:
                    chatbot[num_input_bubbles + index][1] = [None for _ in range(agent_count)]
            for index, rsp in enumerate(display_responses):
                agent_index = self._get_agent_index_by_name(rsp.get(NAME))
                chatbot[num_input_bubbles + index][1][agent_index] = rsp[CONTENT]
            history.extend(final_messages)

        if self.verbose:
            logger.info("agent_run response:\n" + pprint.pformat(final_messages, indent=2))

        return chatbot, history, final_messages

    def _complete_turn(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        title: str,
        agent_selector: int = 0,
    ) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        pre_state = self.fetch_runtime_state()
        chatbot, history, turn_messages = self._run_agent_once(chatbot, history, agent_selector)
        post_state = self.fetch_runtime_state()
        self.record_recent_step(title, pre_state, post_state, turn_messages)
        return chatbot, history, pre_state, post_state

    def single_step(self, chatbot: list[Any] | None, history: list[dict[str, Any]], agent_selector: int = 0):
        self.auto_step_counter += 1
        title = f"自动步骤 {self.auto_step_counter}"
        chatbot, history = self.append_auto_user_turn(chatbot, history, title)
        chatbot, history, _, post_state = self._complete_turn(chatbot, history, title, agent_selector)
        detail = "单步完成"
        stop_reason = self.stop_reason(post_state)
        if stop_reason:
            detail = stop_reason
        yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("空闲", detail)

    def start_autoplay(
        self,
        chatbot: list[Any] | None,
        history: list[dict[str, Any]],
        max_steps: int | float | None,
        max_stuck: int | float | None,
        agent_selector: int = 0,
    ):
        limit = int(max_steps or 0)
        stuck_limit = max(1, int(max_stuck or 6))
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
            chatbot, history = self.append_auto_user_turn(chatbot, history, title)
            yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("运行中", f"{title} 进行中")

            chatbot, history, pre_state, post_state = self._complete_turn(chatbot, history, title, agent_selector)
            pre_sig = self.state_signature(pre_state)
            post_sig = self.state_signature(post_state)
            if pre_sig == post_sig:
                unchanged_rounds += 1
            else:
                unchanged_rounds = 0

            reason = self.stop_reason(post_state)
            if reason:
                self.autoplay_enabled = False
                yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已停止", reason)
                return

            if unchanged_rounds >= stuck_limit:
                self.autoplay_enabled = False
                detail = f"连续 {unchanged_rounds} 轮状态未变化，已暂停"
                yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已停止", detail)
                return

            yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("运行中", f"已执行 {self.auto_step_counter} 步")
            time.sleep(0.2)

        yield chatbot, history, self.render_state_panel(), self.render_recent_steps_html(), self.render_autoplay_status("已暂停", f"已执行 {self.auto_step_counter} 步")

    def pause_autoplay(self):
        self.autoplay_enabled = False
        return self.render_autoplay_status("已暂停", f"已执行 {self.auto_step_counter} 步")

    def change_agent_with_state(self, agent_selector: int):
        return agent_selector, gr.HTML(self.render_agent_info_html(agent_selector))

    def agent_run_with_state(self, chatbot: list[Any] | None, history: list[dict[str, Any]], agent_selector: int = 0):
        title = self._derive_turn_title(history)
        chatbot, history, _, _ = self._complete_turn(chatbot, history, title, agent_selector)
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
        custom_theme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,
            radius_size=gr.themes.utils.sizes.radius_none,
        )

        with gr.Blocks(css=STS2_WEBUI_CSS, theme=custom_theme) as demo:
            history = gr.State(list(messages or []))
            agent_selector = gr.State(0)
            with ms.Application():
                with gr.Row(elem_classes="container"):
                    with gr.Column(scale=16):
                        chatbot = mgr.Chatbot(
                            value=convert_history_to_chatbot(messages=messages),
                            avatar_images=[self.user_config, self.agent_config_list],
                            height=850,
                            avatar_image_width=80,
                            flushing=False,
                            show_copy_button=True,
                            latex_delimiters=[
                                {"left": "\\(", "right": "\\)", "display": True},
                                {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
                                {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
                                {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
                                {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
                                {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
                                {"left": "\\[", "right": "\\]", "display": True},
                            ],
                        )
                        input_box = mgr.MultimodalInput(placeholder=self.input_placeholder)
                        audio_input = gr.Audio(sources=["microphone"], type="filepath")

                    with gr.Column(scale=10, elem_classes="sts2-sidebar"):
                        agent_info_block = gr.HTML(self.render_agent_info_html())
                        state_panel = gr.HTML(self.render_state_panel())
                        recent_steps = gr.HTML(self.render_recent_steps_html())

                        with gr.Row(elem_classes=["sts2-control-row"]):
                            single_step_btn = gr.Button("单步", elem_classes=["sts2-small-btn"])
                            auto_run_btn = gr.Button("自动", variant="primary", elem_classes=["sts2-small-btn"])
                            pause_btn = gr.Button("暂停", variant="stop", elem_classes=["sts2-small-btn"])
                            refresh_btn = gr.Button("刷新", elem_classes=["sts2-small-btn"])

                        with gr.Row(elem_classes=["sts2-control-row"]):
                            max_steps = gr.Number(
                                label="最大步数（0=无限）",
                                value=0,
                                minimum=0,
                                precision=0,
                                elem_classes=["sts2-number-compact"],
                            )
                            max_stuck = gr.Number(
                                label="卡住轮数",
                                value=6,
                                minimum=1,
                                precision=0,
                                elem_classes=["sts2-number-compact"],
                            )

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
                    [chatbot, history, agent_selector],
                    [chatbot, history, state_panel, recent_steps, autoplay_status],
                )
                input_promise.then(self.flushed, None, [input_box])

                single_step_btn.click(
                    fn=self.single_step,
                    inputs=[chatbot, history, agent_selector],
                    outputs=[chatbot, history, state_panel, recent_steps, autoplay_status],
                )
                auto_run_btn.click(
                    fn=self.start_autoplay,
                    inputs=[chatbot, history, max_steps, max_stuck, agent_selector],
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

                if len(self.agent_list) > 1:
                    selector = gr.Dropdown(
                        [(agent.name, i) for i, agent in enumerate(self.agent_list)],
                        label="Agent",
                        value=0,
                        interactive=True,
                    )
                    selector.change(
                        fn=self.change_agent_with_state,
                        inputs=[selector],
                        outputs=[agent_selector, agent_info_block],
                        queue=False,
                    )

            demo.load(None)

        demo.queue(default_concurrency_limit=concurrency_limit).launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
        )


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
                "args": [
                    str(server_script),
                    "--host",
                    args.game_host,
                    "--port",
                    str(args.game_port),
                ],
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

    ui = STS2WebUI(
        bot,
        chatbot_config=chatbot_config,
        game_host=args.game_host,
        game_port=args.game_port,
    )

    print(f"Game API: http://{args.game_host}:{args.game_port}")
    print(f"Qwen-Agent UI: http://{args.host}:{args.port}")
    ui.run(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
