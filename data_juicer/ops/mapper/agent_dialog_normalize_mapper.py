# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Normalize agent interaction format (messages + choices) to DJ fields for
# dialog/text ops. Supports multi-platform, multi-agent tool formats.

import re
from typing import Any, List, Optional, Sequence, Tuple

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "agent_dialog_normalize_mapper"

# Default labels for flattened dialog (i18n / multi-platform)
DEFAULT_USER_LABEL = "用户"
DEFAULT_ASSISTANT_LABEL = "助手"


def _coerce_content_fragment(val: Any) -> str:
    """Turn a content block's text-ish field into a single flat string (no .strip on dict)."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        for k in ("value", "text", "content"):
            if k in val and val[k] not in (None, ""):
                return _coerce_content_fragment(val[k])
        return ""
    if isinstance(val, list):
        return "\n".join(_coerce_content_fragment(x) for x in val if x not in (None, "")).strip()
    return str(val).strip()


def _content_to_text(content: Any) -> str:
    """Extract plain text from message.content.
    Supports: str, list of {type, text} (OpenAI multimodal), list of str.
    ``text`` may be nested (dict/list); Qwen-style ``thinking`` blocks are included.
    """  # noqa: E501
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(_coerce_content_fragment(block.get("text")))
                elif btype == "input_text" or (btype not in ("thinking", "reasoning") and "text" in block):
                    parts.append(_coerce_content_fragment(block.get("text")))
                elif btype in ("thinking", "reasoning") or "thinking" in block or "reasoning_content" in block:
                    # Qwen / DeepSeek / DashScope style reasoning (may omit type)
                    parts.append(
                        _coerce_content_fragment(
                            block.get("thinking") or block.get("reasoning") or block.get("reasoning_content")
                        )
                    )
            elif isinstance(block, str):
                parts.append(block.strip())
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        return _coerce_content_fragment(content)
    return str(content).strip()


def _get_tool_name_from_call(tc: dict) -> Optional[str]:
    """Get tool/function name from one tool_call item.
    Supports OpenAI (function.name), Anthropic (name), generic.
    """
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") or tc.get("function_call")
    if isinstance(fn, dict) and fn.get("name"):
        return fn["name"]
    if tc.get("name"):
        return tc["name"]
    return None


def _tool_calls_summary(
    tool_calls: Any,
    max_names: int = 10,
) -> str:
    """Summarize tool_calls for history; multi-format."""
    if not tool_calls or not isinstance(tool_calls, list):
        return ""
    names = []
    for tc in tool_calls:
        n = _get_tool_name_from_call(tc)
        if n and n not in names:
            names.append(n)
    if not names:
        return ""
    display = names[:max_names]
    if len(names) > max_names:
        display.append(f"...+{len(names) - max_names}")
    return "[Tool calls: " + ", ".join(display) + "]"


def _extract_tool_types(messages: List[dict]) -> List[str]:
    """Collect unique tool/function names from messages (multi-format)."""
    out = []
    seen = set()
    for m in messages:
        for tc in m.get("tool_calls") or m.get("tool_use") or []:
            name = _get_tool_name_from_call(tc)
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


# Skill patterns: only structured identifiers to avoid fragment noise.
# - Path: xxx/SKILL.md or xxx\SKILL.md -> capture segment (ASCII identifier)
# - ## header: only English identifiers (## cron), not doc headings (## 记忆).
_SKILL_PATTERNS = [
    re.compile(r"[/\\]([a-zA-Z][a-zA-Z0-9_]*)[/\\]SKILL\.md", re.IGNORECASE),
    re.compile(r"##\s+([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)?)\s*(?:\n|$)"),
    re.compile(
        r"Check\s+[\"'].*?([a-zA-Z][a-zA-Z0-9_]*)/SKILL\.md",
        re.IGNORECASE,
    ),
]


def _extract_skill_types(messages: List[dict]) -> List[str]:
    """Extract skill names (paths + ## English headers only)."""
    out = []
    seen = set()
    for m in messages:
        text = _content_to_text(m.get("content"))
        for pat in _SKILL_PATTERNS:
            for mo in pat.finditer(text):
                name = (mo.group(1) or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
    return out


def _messages_to_history(
    messages: List[dict],
    include_system_in_first_user: bool = False,
) -> List[Tuple[str, str]]:
    """Convert messages to [(query, response), ...]. User/assistant only.

    Agent/tool note: within one user turn, the model may emit **multiple**
    ``assistant`` messages (tool calls → ``tool`` → assistant again). Earlier
    implementations **replaced** the assistant side each time, dropping
    intermediate reasoning and tool traces. We **accumulate** consecutive
    assistant segments (and still append each ``tool`` result onto the same
    pair) so ``query`` / ``response`` / ``text`` match multi-step agent runs.
    """
    history = []
    pending_system = []

    for m in messages:
        role = (m.get("role") or "").lower()
        content = _content_to_text(m.get("content"))
        tool_calls = m.get("tool_calls") or m.get("tool_use") or []

        if role == "system":
            if include_system_in_first_user and content:
                pending_system.append(content)
            continue
        if role == "user":
            if pending_system:
                content = "\n\n".join(pending_system + [content]).strip()
                pending_system = []
            history.append((content, ""))
            continue
        if role == "assistant":
            if not content and tool_calls:
                content = _tool_calls_summary(tool_calls)
            piece = content or ""
            if history:
                prev_q, prev_r = history[-1]
                if prev_r and piece:
                    new_r = prev_r + "\n\n" + piece
                elif piece:
                    new_r = piece
                else:
                    new_r = prev_r
                history[-1] = (prev_q, new_r)
            else:
                history.append(("", piece))
            continue
        if role == "tool":
            # Append tool result to last assistant response for context
            if history and history[-1][1]:
                history[-1] = (
                    history[-1][0],
                    history[-1][1] + "\n[Tool result]\n" + content[:10000],
                )
            elif history:
                # Rare: tool result before any assistant text (still attach)
                history[-1] = (history[-1][0], "[Tool result]\n" + content[:10000])
            continue
    return history


def _last_user_assistant_msg_indices(
    messages: List[dict],
) -> Tuple[Optional[int], Optional[int]]:
    """0-based indices in ``messages`` of the last user / assistant turns."""
    last_u: Optional[int] = None
    last_a: Optional[int] = None
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").lower()
        if role == "user":
            last_u = i
        elif role == "assistant":
            last_a = i
    return last_u, last_a


def _first_non_empty_str(sample: dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k not in sample:
            continue
        v = sample.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _choices_to_text(choices: Any) -> str:
    """Extract reply text from choices (OpenAI / Anthropic / generic)."""
    if not choices or not isinstance(choices, list):
        return ""
    for c in choices:
        if not isinstance(c, dict):
            continue
        msg = c.get("message") or c.get("delta") or c
        text = msg.get("content")
        if text is None:
            continue
        if isinstance(text, str) and text.strip():
            return text.strip()
        if isinstance(text, list):
            t = _content_to_text(text)
            if t:
                return t
    return ""


def _flatten_history_to_text(
    history: List[Tuple[str, str]],
    user_label: str = DEFAULT_USER_LABEL,
    assistant_label: str = DEFAULT_ASSISTANT_LABEL,
) -> str:
    """Flatten history to one text for text-based ops."""
    lines = []
    for q, r in history:
        if q:
            lines.append(f"{user_label}：{q}")
        if r:
            lines.append(f"{assistant_label}：{r}")
    return "\n\n".join(lines)


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentDialogNormalizeMapper(Mapper):
    """Normalize agent format (messages + choices) to DJ fields.

    Outputs: text, dialog_history, query, response; optionally meta tags
    agent_tool_types, agent_skill_types, agent_turn_count. When
    ``copy_lineage_fields`` is True, also copies request_model, pt,
    total_cost_time, and (when ``copy_request_id``) the first non-empty
    id among ``request_id_keys`` from the sample root into meta for cohort
    analysis and stable drill-down links. Always records last user/assistant
    message indices (in the raw ``messages`` list) when present.
    Supports multi-format tool_calls (e.g. tool_calls[].function.name as in
    OpenAI / demos/local/demo-agent-data-content.json) and configurable
    user/assistant labels.
    """

    def __init__(
        self,
        messages_key: str = "messages",
        choices_key: str = "choices",
        text_key: str = "text",
        history_key: str = "dialog_history",
        query_key: str = "query",
        response_key: str = "response",
        extract_tool_skill_tags: bool = True,
        include_system_in_first_user: bool = False,
        user_label: str = DEFAULT_USER_LABEL,
        assistant_label: str = DEFAULT_ASSISTANT_LABEL,
        copy_lineage_fields: bool = True,
        copy_request_id: bool = True,
        request_id_keys: Tuple[str, ...] = (
            "request_id",
            "trace_id",
            "id",
        ),
        **kwargs,
    ):
        super().__init__(text_key=text_key, **kwargs)
        self.messages_key = messages_key
        self.choices_key = choices_key
        self.history_key = history_key
        self.query_key = query_key
        self.response_key = response_key
        self.extract_tool_skill_tags = extract_tool_skill_tags
        self.include_system_in_first_user = include_system_in_first_user
        self.user_label = user_label
        self.assistant_label = assistant_label
        self.copy_lineage_fields = copy_lineage_fields
        self.copy_request_id = copy_request_id
        self.request_id_keys = request_id_keys

    def process_single(self, sample):
        messages = sample.get(self.messages_key) or []
        choices = sample.get(self.choices_key) or []

        if not isinstance(messages, list):
            messages = []

        history = _messages_to_history(
            messages,
            include_system_in_first_user=self.include_system_in_first_user,
        )
        flat_text = _flatten_history_to_text(
            history,
            user_label=self.user_label,
            assistant_label=self.assistant_label,
        )
        last_query = history[-1][0] if history else ""
        last_response = history[-1][1] if history else ""
        if not last_response and choices:
            last_response = _choices_to_text(choices)

        sample[self.text_key] = flat_text
        sample[self.history_key] = history
        sample[self.query_key] = last_query
        sample[self.response_key] = last_response

        if Fields.meta not in sample:
            sample[Fields.meta] = {}
        meta = sample[Fields.meta]
        meta[MetaKeys.agent_turn_count] = len(history)
        if self.extract_tool_skill_tags:
            meta[MetaKeys.agent_tool_types] = _extract_tool_types(messages)
            meta[MetaKeys.agent_skill_types] = _extract_skill_types(messages)

        last_u_idx, last_a_idx = _last_user_assistant_msg_indices(messages)
        if last_u_idx is not None:
            meta[MetaKeys.agent_last_user_msg_idx] = last_u_idx
        if last_a_idx is not None:
            meta[MetaKeys.agent_last_assistant_msg_idx] = last_a_idx
        if self.copy_request_id:
            rid = _first_non_empty_str(sample, self.request_id_keys)
            if rid is not None:
                meta[MetaKeys.agent_request_id] = rid

        # Cohort fields for bad-case / A-B analysis (request_model, date bucket, latency)
        if self.copy_lineage_fields:
            if sample.get("request_model") is not None:
                meta[MetaKeys.agent_request_model] = sample["request_model"]
            if sample.get("pt") is not None:
                meta[MetaKeys.agent_pt] = sample["pt"]
            if sample.get("total_cost_time") is not None:
                meta[MetaKeys.agent_total_cost_time_ms] = sample["total_cost_time"]

        return sample
