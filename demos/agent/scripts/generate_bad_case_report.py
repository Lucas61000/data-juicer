#!/usr/bin/env python3
"""HTML report: bad-case tiers, signals, cohort table, embedded charts.

Merges ``*_stats.jsonl`` with ``processed.jsonl`` when needed.

Example:
  python demos/agent/scripts/generate_bad_case_report.py \\
    --input ./outputs/agent_quality/processed.jsonl \\
    --output ./outputs/agent_quality/bad_case_report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_bad_case_cohorts import aggregate_cohort_stdlib, load_merged_rows  # noqa: E402
from bad_case_signal_support import SIGNAL_SUPPORT_ROWS  # noqa: E402
from dj_export_row import get_dj_meta  # noqa: E402


def _get_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:  # pragma: no cover
        return None


def _fig_to_data_uri(fig, plt_mod) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt_mod.close(fig)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _tier_counts(rows: List[dict]) -> Counter:
    c: Counter = Counter()
    for row in rows:
        meta = get_dj_meta(row)
        c[str(meta.get("agent_bad_case_tier", "none"))] += 1
    return c


def _signal_counts_by_weight(rows: List[dict]) -> Tuple[Counter, Counter]:
    high_c: Counter = Counter()
    med_c: Counter = Counter()
    for row in rows:
        meta = get_dj_meta(row)
        for s in meta.get("agent_bad_case_signals") or []:
            if not isinstance(s, dict) or not s.get("code"):
                continue
            code = str(s["code"])
            w = s.get("weight")
            if w == "high":
                high_c[code] += 1
            elif w == "medium":
                med_c[code] += 1
    return high_c, med_c


def _attribution_table_html() -> str:
    parts = []
    for r in SIGNAL_SUPPORT_ROWS:
        role = "主证据" if r["role"] == "primary" else "附录·启发式"
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(r['code'])}</code></td>"
            f"<td>{html.escape(role)}</td>"
            f"<td>{html.escape(str(r['weight_hint']))}</td>"
            f"<td>{html.escape(str(r['upstream']))}</td>"
            "</tr>"
        )
    thead = (
        "<thead><tr><th>signal code</th><th>角色</th><th>典型权重</th>"
        "<th>上游字段与算子</th></tr></thead>"
    )
    return f"<table>{thead}<tbody>{''.join(parts)}</tbody></table>"


def _model_tier_matrix(rows: List[dict]) -> Dict[str, Counter]:
    m: DefaultDict[str, Counter] = defaultdict(Counter)
    for row in rows:
        meta = get_dj_meta(row)
        model = str(meta.get("agent_request_model") or "_unknown")
        tier = str(meta.get("agent_bad_case_tier", "none"))
        m[model][tier] += 1
    return dict(m)


def _json_pretty(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except TypeError:  # pragma: no cover
        return str(obj)


def _collect_drilldown(rows: List[dict], limit: int) -> List[dict]:
    """Bad-case rows (high_precision, watchlist) with stable ids for the report UI."""
    tier_rank = {"high_precision": 0, "watchlist": 1}
    scored: List[Tuple[int, int, dict]] = []
    for i, row in enumerate(rows):
        meta = get_dj_meta(row)
        tier = str(meta.get("agent_bad_case_tier", "none"))
        if tier not in tier_rank:
            continue
        scored.append((tier_rank[tier], i, row))
    scored.sort(key=lambda x: (x[0], x[1]))
    out: List[dict] = []
    for _tr, _orig_i, row in scored[:limit]:
        meta = get_dj_meta(row)
        tier = str(meta.get("agent_bad_case_tier", ""))
        rid = (
            meta.get("agent_request_id")
            or row.get("request_id")
            or row.get("trace_id")
            or row.get("id")
        )
        rid_s = str(rid).strip() if rid is not None else ""
        u_idx = meta.get("agent_last_user_msg_idx")
        a_idx = meta.get("agent_last_assistant_msg_idx")
        signals = meta.get("agent_bad_case_signals") or []
        insight = meta.get("agent_insight_llm") or {}
        meta_subset = {
            k: meta[k]
            for k in (
                "agent_request_id",
                "agent_last_user_msg_idx",
                "agent_last_assistant_msg_idx",
                "agent_request_model",
                "agent_pt",
                "agent_bad_case_tier",
                "agent_turn_count",
                "agent_total_cost_time_ms",
            )
            if k in meta
        }
        out.append(
            {
                "tier": tier,
                "request_id": rid_s,
                "u_idx": u_idx,
                "a_idx": a_idx,
                "model": str(meta.get("agent_request_model") or ""),
                "pt": str(meta.get("agent_pt") or ""),
                "query": row.get("query") or "",
                "response": row.get("response") or "",
                "signals_json": _json_pretty(signals),
                "insight_json": _json_pretty(insight),
                "meta_json": _json_pretty(meta_subset),
            }
        )
    return out


def _idx_badge(u_idx: object, a_idx: object) -> str:
    parts = []
    if u_idx is not None:
        parts.append(f"user_idx={u_idx}")
    if a_idx is not None:
        parts.append(f"asst_idx={a_idx}")
    return " · ".join(parts) if parts else "—"


def _drilldown_section_html(drill: List[dict]) -> str:
    if not drill:
        return (
            "<h2>样本钻取（high / watchlist）</h2>"
            "<p class='note'>本批无 <code>high_precision</code> / "
            "<code>watchlist</code> 样本，或已将钻取条数上限设为 0。</p>"
        )
    cards = []
    for i, d in enumerate(drill):
        anchor = f"bc-drill-{i}"
        tier = html.escape(d["tier"])
        tier_cls = "tier-hp" if d["tier"] == "high_precision" else "tier-wl"
        rid = html.escape(d["request_id"] or "—")
        idx_txt = html.escape(_idx_badge(d["u_idx"], d["a_idx"]))
        model = html.escape(d["model"] or "—")
        pt = html.escape(d["pt"] or "—")
        cards.append(
            f'<div class="drill-card" id="{anchor}">'
            '<div class="drill-summary">'
            f'<span class="tier-tag {tier_cls}">{tier}</span> '
            f'<a class="anchor-link" href="#{anchor}" title="锚点">#{i}</a> '
            f"<code class=\"rid\" title=\"request_id / trace / id\">{rid}</code> "
            f'<span class="idx" title="messages 中下标（0-based）">{idx_txt}</span> '
            f'<span class="cohort-mini">{model} · {pt}</span> '
            '<button type="button" class="drill-toggle" aria-expanded="false">'
            "展开字段</button>"
            "</div>"
            '<div class="drill-body" hidden>'
            '<div class="field-grid">'
            "<section><h4>query</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['query'])}</pre></section>"
            "<section><h4>response</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['response'])}</pre></section>"
            "<section><h4>agent_bad_case_signals</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['signals_json'])}</pre></section>"
            "<section><h4>agent_insight_llm</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['insight_json'])}</pre></section>"
            "<section><h4>meta（钻取子集）</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['meta_json'])}</pre></section>"
            "</div></div></div>"
        )
    block = (
        "<h2>样本钻取（high / watchlist）</h2>"
        "<p class='note'>每条可展开查看 <code>query</code>、<code>response</code>、"
        "结构化信号与 LLM insight；表头含 <strong>request_id</strong>（及后备 "
        "<code>trace_id</code> / <code>id</code>）与 "
        "<strong>messages</strong> 中最后 user/assistant 的下标，便于对齐日志与展陈。"
        "<code>#n</code> 为页内锚点，可复制浏览器地址栏 fragment 分享。</p>"
        '<div class="drill-list">'
        f"{''.join(cards)}</div>"
    )
    return block


def _insight_samples(
    rows: List[dict],
    tier: str,
    limit: int,
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for row in rows:
        if len(out) >= limit:
            break
        meta = get_dj_meta(row)
        if str(meta.get("agent_bad_case_tier", "")) != tier:
            continue
        ins = meta.get("agent_insight_llm") or {}
        hl = (ins.get("headline") or "").strip()
        if hl:
            out.append((hl, str(meta.get("agent_request_model", ""))))
    return out


def _chart_tier_bar(tier_cnt: Counter, plt_mod) -> Optional[str]:
    if plt_mod is None:
        return None
    order = ("high_precision", "watchlist", "none")
    labels = [t for t in order if tier_cnt.get(t, 0) > 0]
    if not labels:
        labels = list(tier_cnt.keys())
    vals = [tier_cnt[t] for t in labels]
    fig, ax = plt_mod.subplots(figsize=(6, 3.2))
    colors = {
        "high_precision": "#c0392b",
        "watchlist": "#f39c12",
        "none": "#95a5a6",
    }
    ax.bar(
        labels,
        vals,
        color=[colors.get(x, "#3498db") for x in labels],
    )
    ax.set_title("Bad-case tier counts")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _chart_signals(
    sig_cnt: Counter,
    plt_mod,
    title: str,
    color: str = "#2980b9",
    top_n: int = 14,
) -> Optional[str]:
    if plt_mod is None or not sig_cnt:
        return None
    items = sig_cnt.most_common(top_n)
    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    fig, ax = plt_mod.subplots(figsize=(7, max(2.5, 0.35 * len(labels))))
    ax.barh(labels[::-1], vals[::-1], color=color)
    ax.set_title(title)
    ax.set_xlabel("Count")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _chart_by_model(model_tier: Dict[str, Counter], plt_mod) -> Optional[str]:
    if plt_mod is None or len(model_tier) == 0:
        return None
    models = sorted(model_tier.keys())
    tiers = ("high_precision", "watchlist", "none")
    fig, ax = plt_mod.subplots(figsize=(max(6, len(models) * 0.9), 3.8))
    bottom = [0] * len(models)
    colors = {"high_precision": "#c0392b", "watchlist": "#f39c12", "none": "#bdc3c7"}
    for tier in tiers:
        vs = [model_tier[m].get(tier, 0) for m in models]
        if not any(vs):
            continue
        ax.bar(models, vs, bottom=bottom, label=tier, color=colors[tier], width=0.65)
        bottom = [b + v for b, v in zip(bottom, vs)]
    ax.set_title("Tier mix by agent_request_model")
    ax.set_ylabel("Samples")
    ax.legend()
    plt_mod.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _html_page(
    title: str,
    input_path: str,
    n_rows: int,
    tier_cnt: Counter,
    cohort_rows: List[dict],
    chart_tier: Optional[str],
    chart_model: Optional[str],
    chart_sig_high: Optional[str],
    chart_sig_med: Optional[str],
    attribution_table: str,
    samples_hp: List[Tuple[str, str]],
    drilldown_html: str,
) -> str:
    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tier_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(tier_cnt.items(), key=lambda x: -x[1])
    )
    cohort_lines = []
    for r in cohort_rows:
        if not r.get("count") and not r.get("top_signal_codes"):
            continue
        cohort_lines.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('agent_request_model', '')))}</td>"
            f"<td>{html.escape(str(r.get('agent_pt', '')))}</td>"
            f"<td>{html.escape(str(r.get('tier', '')))}</td>"
            f"<td>{int(r.get('count') or 0)}</td>"
            f"<td>{html.escape(str(r.get('top_signal_codes', '')))}</td>"
            "</tr>"
        )
    sample_block = ""
    if samples_hp:
        parts = []
        for h, m in samples_hp:
            parts.append(
                "<li><span class='model'>"
                f"{html.escape(m)}</span> — {html.escape(h)}</li>"
            )
        lis = "".join(parts)
        sample_block = (
            "<h2>Insight headlines (high_precision sample)</h2><ul>"
            f"{lis}</ul>"
        )

    charts = []
    if chart_tier:
        charts.append(
            "<h2>Tier overview</h2>"
            f"<img src='{chart_tier}' alt='tiers'/>"
        )
    if chart_model:
        charts.append(
            "<h2>By request model</h2>"
            f"<img src='{chart_model}' alt='by model'/>"
        )
    if chart_sig_high:
        charts.append(
            "<h3>本批次 high 权重信号</h3>"
            f"<img src='{chart_sig_high}' alt='signals high'/>"
        )
    if chart_sig_med:
        charts.append(
            "<h3>附录：medium 启发式信号</h3>"
            "<p class='note'>多为单条弱证据；tier 需与其它信号组合才可能进 watchlist。</p>"
            f"<img src='{chart_sig_med}' alt='signals medium'/>"
        )
    if charts:
        charts_html = "\n".join(charts)
    else:
        charts_html = "<p>(Charts skipped — no matplotlib)</p>"

    css = (
        "body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;}"
        "h1{font-size:1.35rem;}.meta{color:#555;font-size:0.9rem;}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;}"
        "th{background:#f4f4f4;}img{max-width:100%;height:auto;}"
        "ul{line-height:1.5;}.model{color:#555;font-size:0.85rem;}"
        "details{margin-top:2rem;padding:12px;background:#fafafa;border:1px solid #eee;}"
        "summary{cursor:pointer;font-weight:600;}"
        ".note{color:#666;font-size:0.88rem;}"
        ".drill-list{display:flex;flex-direction:column;gap:10px;margin:1rem 0;}"
        ".drill-card{border:1px solid #ddd;border-radius:8px;background:#fff;"
        "box-shadow:0 1px 2px rgba(0,0,0,0.04);}"
        ".drill-summary{display:flex;flex-wrap:wrap;align-items:center;gap:8px;"
        "padding:10px 12px;background:#fafafa;border-radius:8px 8px 0 0;"
        "border-bottom:1px solid #eee;}"
        ".drill-body{padding:12px;border-radius:0 0 8px 8px;}"
        ".tier-tag{font-size:0.75rem;padding:2px 8px;border-radius:999px;"
        "font-weight:600;}"
        ".tier-hp{background:#fadbd8;color:#922b21;}"
        ".tier-wl{background:#fdebd0;color:#9c640c;}"
        ".rid{font-size:0.85rem;background:#f4f6f8;padding:2px 6px;border-radius:4px;}"
        ".idx{font-size:0.82rem;color:#555;}"
        ".cohort-mini{font-size:0.82rem;color:#666;}"
        ".anchor-link{font-weight:600;margin-right:4px;}"
        "button.drill-toggle{margin-left:auto;font-size:0.85rem;cursor:pointer;"
        "padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#fff;}"
        "button.drill-toggle:hover{background:#f0f0f0;}"
        ".field-grid{display:grid;gap:14px;}"
        "@media(min-width:900px){.field-grid{grid-template-columns:1fr 1fr;}}"
        ".field-grid section{margin:0;}"
        ".field-grid h4{margin:0 0 6px;font-size:0.9rem;color:#333;}"
        "pre.drill-pre{margin:0;white-space:pre-wrap;word-break:break-word;"
        "max-height:320px;overflow:auto;font-size:0.82rem;line-height:1.45;"
        "background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:6px;}"
    )
    thead = (
        "<thead><tr><th>agent_request_model</th><th>agent_pt</th>"
        "<th>tier</th><th>count</th><th>top_signal_codes</th></tr></thead>"
    )
    adv = (
        "<p>进阶说明见 <code>demos/agent/BAD_CASE_INSIGHTS.md</code>、"
        "<code>ENTITY_RELATION_TUNING.md</code>、"
        "<code>demos/agent/scripts/README.md</code>。</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{css}</style>
</head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">
  Generated {html.escape(gen_at)} ·
  Input: <code>{html.escape(input_path)}</code><br/>
  Rows: <strong>{n_rows}</strong>
</div>
<h2>Tier counts</h2>
<table><thead><tr><th>Tier</th><th>Count</th></tr></thead>
<tbody>{tier_rows}</tbody></table>
<h2>Bad-case mining · 归因链</h2>
<p>下列说明各 <code>agent_bad_case_signals[].code</code> 依赖的 <strong>meta / stats</strong> 字段
及典型算子来源（自明支撑 bad case mining）。</p>
{attribution_table}
{drilldown_html}
<h2>Charts</h2>
{charts_html}
<h2>Cohort detail (model × pt × tier)</h2>
<table>{thead}<tbody>{"".join(cohort_lines)}</tbody></table>
{sample_block}
<details>
<summary>Advanced / debugging</summary>
{adv}
</details>
<script>
(function () {{
  document.querySelectorAll("button.drill-toggle").forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      var card = btn.closest(".drill-card");
      if (!card) return;
      var body = card.querySelector(".drill-body");
      if (!body) return;
      var open = body.hasAttribute("hidden");
      if (open) {{
        body.removeAttribute("hidden");
        btn.textContent = "收起字段";
        btn.setAttribute("aria-expanded", "true");
      }} else {{
        body.setAttribute("hidden", "");
        btn.textContent = "展开字段";
        btn.setAttribute("aria-expanded", "false");
      }}
    }});
  }});
}})();
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="processed.jsonl")
    ap.add_argument("--output", required=True, help="Output .html path")
    ap.add_argument(
        "--title",
        default="Agent bad-case report",
        help="HTML title / H1",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max rows to read")
    ap.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip matplotlib figures (table-only HTML)",
    )
    ap.add_argument(
        "--sample-headlines",
        type=int,
        default=8,
        help="Max high_precision insight headlines to list (0=off)",
    )
    ap.add_argument(
        "--drilldown-limit",
        type=int,
        default=48,
        help="Max high/watchlist samples with expandable field drill-down (0=skip section)",
    )
    args = ap.parse_args()

    rows = load_merged_rows(args.input, args.limit)
    if not rows:
        print("ERROR: no rows loaded; check --input path.", file=sys.stderr)
        return 2

    tier_cnt = _tier_counts(rows)
    high_c, med_c = _signal_counts_by_weight(rows)
    model_tier = _model_tier_matrix(rows)
    cohort = aggregate_cohort_stdlib(rows)
    att_html = _attribution_table_html()

    chart_tier = chart_sig_high = chart_sig_med = chart_model = None
    plt_mod = None
    if not args.no_charts:
        plt_mod = _get_plt()
    if plt_mod is not None:
        chart_tier = _chart_tier_bar(tier_cnt, plt_mod)
        if high_c:
            chart_sig_high = _chart_signals(
                high_c,
                plt_mod,
                "High-weight signals (this batch)",
                "#c0392b",
            )
        if med_c:
            chart_sig_med = _chart_signals(
                med_c,
                plt_mod,
                "Appendix: medium / heuristic signals",
                "#2980b9",
            )
        if len(model_tier) >= 1:
            chart_model = _chart_by_model(model_tier, plt_mod)

    samples: List[Tuple[str, str]] = []
    if args.sample_headlines > 0:
        samples = _insight_samples(rows, "high_precision", args.sample_headlines)

    drill_html = ""
    if args.drilldown_limit > 0:
        drill_rows = _collect_drilldown(rows, args.drilldown_limit)
        drill_html = _drilldown_section_html(drill_rows)

    page = _html_page(
        args.title,
        args.input,
        len(rows),
        tier_cnt,
        cohort,
        chart_tier,
        chart_model,
        chart_sig_high,
        chart_sig_med,
        att_html,
        samples,
        drill_html,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
