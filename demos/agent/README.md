# Agent quality & bad-case docs

**English:** This folder documents a **recipe-driven** pipeline for agent logs
(JSONL with `messages` + `choices` / `response_choices`). Core operators live
under `data_juicer/ops/mapper/` (e.g. `agent_dialog_normalize_mapper`,
`agent_bad_case_signal_mapper`, `dialog_*` quality mappers); default prompts
and flattened dialog labels are **English-first** for upstream reuse—override
`user_label` / `assistant_label` on the normalize mapper or subclass mappers
for other locales.

---

# Agent 质检 / bad-case 文档索引

| 需求 | 文档 |
|------|------|
| **只出 HTML 报告**（已有 `processed.jsonl`） | [`BAD_CASE_REPORT.md`](BAD_CASE_REPORT.md) |
| **怎么跑 smoke / full / 校准 / 常见问题** | [`QUICKSTART_BAD_CASE.md`](QUICKSTART_BAD_CASE.md) |
| **分层逻辑、信号表、insight 字段、jq** | [`BAD_CASE_INSIGHTS.md`](BAD_CASE_INSIGHTS.md) |
| **后处理 Python 脚本参数** | [`scripts/README.md`](scripts/README.md) |
| **LLM 算子加速** | [`PERFORMANCE_LLM.md`](PERFORMANCE_LLM.md) |
| **KenLM / perplexity（macOS）** | [`KENLM_MACOS.md`](KENLM_MACOS.md) |
| **实体·关系算子调参** | [`ENTITY_RELATION_TUNING.md`](ENTITY_RELATION_TUNING.md) |

端到端配置：**`agent_interaction_quality_analysis.yaml`**；无 API 冒烟：**`minimal_configs/09_bad_case_smoke.yaml`**。
