"""Unit tests for AgentInsightLLMMapper (no live API)."""

import unittest

from data_juicer.ops.mapper.agent_insight_llm_mapper import AgentInsightLLMMapper
from data_juicer.utils.constant import Fields, MetaKeys


class TestAgentInsightLLMMapper(unittest.TestCase):

    def test_skips_when_tier_not_in_run_for_tiers(self):
        op = AgentInsightLLMMapper(api_model="gpt-4o", run_for_tiers=["watchlist"])
        sample = {
            Fields.meta: {MetaKeys.agent_bad_case_tier: "none"},
            Fields.stats: {},
            "query": "q",
            "response": "r",
        }
        out = op.process_single(sample)
        self.assertNotIn(MetaKeys.agent_insight_llm, out[Fields.meta])


if __name__ == "__main__":
    unittest.main()
