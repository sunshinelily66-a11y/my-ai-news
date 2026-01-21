import os
import json
import datetime
from zoneinfo import ZoneInfo

import requests
import feedparser
from dateutil import parser as dtparser


FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# 起步源，偏公司动态与产品发布
RSS_FEEDS = [
    "https://huggingface.co/blog/feed.xml",
    "https://www.microsoft.com/en-us/research/blog/feed/",
    "https://arxiv.org/rss/cs.AI",
    # Google The Keyword 的 RSS 总入口，页面有 RSS 入口指向 https://blog.google/rss/ :contentReference[oaicite:8]{index=8}
    "https://blog.google/rss/",
    # NVIDIA 新闻 RSS 聚合入口，可替换为更细分类 RSS :contentReference[oaicite:9]{index=9}
    "https://nvidianews.nvidia.com/rss",
]

MAX_ITEMS_TOTAL = 30


def _safe_parse_time(s: str):
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None


def fetch_recent_items(hours=24, max_items=MAX_ITEMS_TOTAL):
    now = datetime.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    cutoff = now - datetime.timedelta(hours=hours)

    items = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for e in getattr(feed, "entries", [])[: 50]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()

            published = getattr(e, "published", "") or getattr(e, "updated", "")
            published_dt = _safe_parse_time(published)
            if published_dt and published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Shanghai"))
            elif published_dt:
                published_dt = published_dt.astimezone(ZoneInfo("Asia/Shanghai"))

            summary = getattr(e, "summary", "") or getattr(e, "description", "")

            if not title or not link:
                continue
            if published_dt and published_dt < cutoff:
                continue

            items.append(
                {
                    "title": title,
                    "url": link,
                    "published": published_dt.isoformat() if published_dt else published,
                    "summary": " ".join(summary.replace("\n", " ").split())[:300],
                    "source": getattr(feed.feed, "title", "") or feed_url,
                }
            )

    # URL 去重
    seen = set()
    deduped = []
    for it in items:
        key = it["url"].split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    # 按发布时间倒序
    def sort_key(x):
        t = _safe_parse_time(x.get("published", ""))
        if not t:
            return datetime.datetime(1970, 1, 1, tzinfo=ZoneInfo("UTC"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=ZoneInfo("UTC"))
        return t

    deduped.sort(key=sort_key, reverse=True)
    return deduped[:max_items]


def build_input_text(items):
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. [{it['source']}] {it['title']}\n"
            f"URL: {it['url']}\n"
            f"Time: {it['published']}\n"
            f"Snippet: {it['summary']}\n"
        )
    return "\n".join(lines)


def call_deepseek_digest(input_text: str):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions"  # OpenAI compatible :contentReference[oaicite:10]{index=10}
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    schema_hint = {
        "date": "YYYY-MM-DD",
        "top5": [
            {
                "title": "string",
                "one_liner": "string",
                "why_it_matters": "string",
                "winners_losers": "string",
                "confidence": "high|medium|low",
                "urls": ["url"]
            }
        ],
        "company_updates": [
            {
                "company": "string",
                "update": "string",
                "implication": "string",
                "confidence": "high|medium|low",
                "urls": ["url"]
            }
        ],
        "product_releases": [
            {
                "company": "string",
                "product": "string",
                "what_changed": "string",
                "who_cares": "string",
                "adoption_barriers": ["string"],
                "confidence": "high|medium|low",
                "urls": ["url"]
            }
        ],
        "watchlist": [
            {
                "item": "string",
                "metric": "string",
                "timeframe": "7d|14d|30d|this_week",
                "why": "string"
            }
        ],
        "sources": ["url"]
    }

    prompt = f"""
你将基于输入材料生成一份 AI 日报，重点关注公司动态与产品发布。
要求：
1) 事实只能来自输入材料，不要编造。每条结论给 urls。
2) 推断要写清楚，并降低 confidence。
3) 同一事件多条来源合并，urls 保留 1 到 3 个。
4) 输出必须是严格 JSON，结构参考 schema_hint。
schema_hint:
{json.dumps(schema_hint, ensure_ascii=False)}

输入材料：
{input_text}
""".strip()

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        # DeepSeek 提供 JSON Output 能力，确保严格 JSON :contentReference[oaicite:11]{index=11}
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def render_text(digest: dict):
    def fmt_list(items, fn):
        return "\n".join([fn(x, idx) for idx, x in enumerate(items, 1)]) if items else "无"

    date = digest.get("date", "")
    top5 = digest.get("top5", [])
    company_updates = digest.get("company_updates", [])
    product_releases = digest.get("product_releases", [])
    watchlist = digest.get("watchlist", [])
    sources = digest.get("sources", [])

    top5_text = fmt_list(top5, lambda x, i: (
        f"{i}) {x.get('title','')}\n"
        f"   {x.get('one_liner','')}\n"
        f"   Why: {x.get('why_it_matters','')}\n"
        f"   Winners/Losers: {x.get('winners_losers','')}\n"
        f"   Confidence: {x.get('confidence','')}\n"
        f"   Links: " + ", ".join(x.get("urls", [])[:3])
    ))

    cu_text = fmt_list(company_updates, lambda x, i: (
        f"{i}) {x.get('company','')}\n"
        f"   Update: {x.get('update','')}\n"
        f"   Implication: {x.get('implication','')}\n"
        f"   Confidence: {x.get('confidence','')}\n"
        f"   Links: " + ", ".join(x.get("urls", [])[:3])
    ))

    pr_text = fmt_list(product_releases, lambda x, i: (
        f"{i}) {x.get('company','')} | {x.get('product','')}\n"
        f"   Change: {x.get('what_changed','')}\n"
        f"   Who cares: {x.get('who_cares','')}\n"
        f"   Barriers: " + "; ".join(x.get("adoption_barriers", [])[:4]) + "\n"
        f"   Confidence: {x.get('confidence','')}\n"
        f"   Links: " + ", ".join(x.get("urls", [])[:3])
    ))

    wl_text = fmt_list(watchlist, lambda x, i: (
        f"{i}) {x.get('item','')} | {x.get('metric','')} | {x.get('timeframe','')}\n"
        f"   Why: {x.get('why','')}"
    ))

    src_text = "\n".join([f"- {u}" for u in sources[:20]]) if sources else "无"

    msg = (
        f"AI Daily Digest | {date}\n\n"
        f"Top 5\n{top5_text}\n\n"
        f"公司动态\n{cu_text}\n\n"
        f"产品发布\n{pr_text}\n\n"
        f"Watchlist\n{wl_text}\n\n"
        f"Sources\n{src_text}\n"
    )
    return msg[:18000]


def send_feishu_text(text: str):
    if not FEISHU_WEBHOOK:
        raise RuntimeError("Missing FEISHU_WEBHOOK")
    payload = {"msg_type": "text", "content": {"text": text}}
    r = requests.post(FEISHU_WEBHOOK, json=payload, timeout=30)
    r.raise_for_status()


def main():
    now_bj = datetime.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    items = fetch_recent_items(hours=24, max_items=MAX_ITEMS_TOTAL)
    if not items:
        send_feishu_text(f"AI Daily Digest | {now_bj.date().isoformat()}\n\n过去 24 小时未抓取到条目。")
        return

    input_text = build_input_text(items)
    digest = call_deepseek_digest(input_text)

    if not digest.get("date"):
        digest["date"] = now_bj.date().isoformat()

    msg = render_text(digest)
    send_feishu_text(msg)


if __name__ == "__main__":
    main()
