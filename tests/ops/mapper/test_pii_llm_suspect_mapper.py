# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from data_juicer.ops.mapper.pii_llm_suspect_mapper import (
    DEFAULT_REDACTION_PLACEHOLDER,
    PiiLlmSuspectMapper,
    _extract_json_object,
    _heuristic_trigger,
)
from data_juicer.utils.constant import Fields, MetaKeys


class TestPiiLlmSuspectHelpers(unittest.TestCase):
    def test_heuristic_long_digits(self):
        self.assertTrue(_heuristic_trigger("call me 13812345678"))

    def test_heuristic_email_at(self):
        self.assertTrue(_heuristic_trigger("not redacted a@b.co"))

    def test_heuristic_secret_keyword(self):
        self.assertTrue(_heuristic_trigger("api_key: something"))

    def test_heuristic_negative(self):
        self.assertFalse(_heuristic_trigger("hello world only"))

    def test_extract_json_object(self):
        raw = 'prefix {"suspected": [], "likely_clean": true} suffix'
        obj = _extract_json_object(raw)
        self.assertIsNotNone(obj)
        self.assertEqual(obj["suspected"], [])
        self.assertTrue(obj["likely_clean"])

    def test_extract_json_fenced(self):
        raw = '```json\n{"a": 1}\n```'
        obj = _extract_json_object(raw)
        self.assertEqual(obj, {"a": 1})


class TestPiiLlmSuspectMapper(unittest.TestCase):
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_heuristic_gate_skips(self, mock_get, mock_prepare):
        mock_prepare.return_value = lambda device="cpu": None

        m = PiiLlmSuspectMapper(api_model="qwen-turbo", gate_mode="heuristic")
        sample = {Fields.meta: {}, "text": "short"}
        out = m.process_single(sample)
        self.assertFalse(mock_get.called)
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertTrue(meta.get("skipped"))
        self.assertEqual(meta.get("reason"), "heuristic_gate")

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_llm_writes_meta(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return (
                '{"suspected":[{"field":"query","category":"phone",'
                '"evidence":"138****"}], "likely_clean": false}'
            )

        mock_prepare.return_value = lambda device="cpu": fake_client
        mock_get.return_value = fake_client

        m = PiiLlmSuspectMapper(api_model="qwen-turbo", gate_mode="always")
        sample = {
            Fields.meta: {},
            "query": "phone 13812345678 here",
        }
        out = m.process_single(sample)
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertEqual(len(meta.get("suspected", [])), 1)
        self.assertEqual(meta["suspected"][0]["field"], "query")
        self.assertFalse(meta.get("likely_clean", True))

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_no_overwrite(self, mock_get, mock_prepare):
        mock_prepare.return_value = lambda device="cpu": None
        existing = {"suspected": [], "likely_clean": True}
        m = PiiLlmSuspectMapper(api_model="qwen-turbo", overwrite=False)
        sample = {
            Fields.meta: {MetaKeys.pii_llm_suspect: existing},
            "query": "13812345678",
        }
        out = m.process_single(sample)
        self.assertIs(out[Fields.meta][MetaKeys.pii_llm_suspect], existing)
        self.assertFalse(mock_get.called)

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_redaction_evidence_substrings(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return (
                '{"suspected":[{"field":"query","category":"phone",'
                '"evidence":"13812345678"}], "likely_clean": false}'
            )

        mock_prepare.return_value = lambda device="cpu": fake_client
        mock_get.return_value = fake_client

        m = PiiLlmSuspectMapper(
            api_model="qwen-turbo",
            gate_mode="always",
            redaction_mode="evidence",
        )
        sample = {
            Fields.meta: {},
            "query": "phone 13812345678 here",
        }
        out = m.process_single(sample)
        self.assertIn(DEFAULT_REDACTION_PLACEHOLDER, out["query"])
        self.assertNotIn("13812345678", out["query"])
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertEqual(meta.get("redaction_mode"), "evidence")
        self.assertTrue(meta.get("redaction_applied"))

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_redaction_whole_field(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return (
                '{"suspected":[{"field":"query","category":"risk",'
                '"evidence":""}], "likely_clean": false}'
            )

        mock_prepare.return_value = lambda device="cpu": fake_client
        mock_get.return_value = fake_client

        m = PiiLlmSuspectMapper(
            api_model="qwen-turbo",
            gate_mode="always",
            redaction_mode="whole_field",
        )
        sample = {
            Fields.meta: {},
            "query": "sensitive entire field",
        }
        out = m.process_single(sample)
        self.assertEqual(out["query"], DEFAULT_REDACTION_PLACEHOLDER)


if __name__ == "__main__":
    unittest.main()
