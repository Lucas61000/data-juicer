# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent_dialog_normalize_mapper multi-step tool / assistant chains."""

from data_juicer.ops.mapper.agent_dialog_normalize_mapper import _messages_to_history


def test_messages_to_history_accumulates_multiple_assistant_turns():
    """Same user turn: assistant → tool → assistant must not drop the first leg."""
    messages = [
        {"role": "user", "content": "List the folder"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "ls", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "Error: not found"},
        {"role": "assistant", "content": "Retrying with another path."},
        {"role": "tool", "content": "file.txt"},
        {"role": "assistant", "content": "Found file.txt."},
    ]
    history = _messages_to_history(messages)
    assert len(history) == 1
    q, r = history[0]
    assert q == "List the folder"
    assert "[Tool calls:" in r
    assert "Retrying with another path." in r
    assert "Found file.txt." in r
    assert "[Tool result]" in r
    assert "not found" in r
