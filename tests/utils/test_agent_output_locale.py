# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.utils.agent_output_locale import (
    dialog_score_json_instruction,
    llm_filter_free_text_language_appendix,
    normalize_preferred_output_lang,
)


class TestAgentOutputLocale(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_preferred_output_lang("zh-CN"), "zh")
        self.assertEqual(normalize_preferred_output_lang("EN"), "en")
        self.assertEqual(normalize_preferred_output_lang(None), "en")

    def test_json_instruction_zh_has_score(self):
        s = dialog_score_json_instruction("zh")
        self.assertIn("score", s)
        self.assertIn("reason", s)

    def test_filter_appendix_empty_when_none(self):
        self.assertEqual(llm_filter_free_text_language_appendix(None), "")


if __name__ == "__main__":
    unittest.main()
