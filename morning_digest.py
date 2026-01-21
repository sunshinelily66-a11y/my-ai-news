import os
import json
import datetime
from zoneinfo import ZoneInfo

import requests
import feedparser
from dateutil import parser as dtparser


FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# 48h 时效窗口
RECENT_HOURS = 48
MAX_ITEMS_TOTAL = 45  # 条目多一点，后面交给模型做合并与筛选

# 更丰富的信息源：公司动态 + 产品发布 + 商业咨询/投融资
# 说明：尽量选英文主流来源，避免中文站点依赖
RSS_FEEDS = [
    # 产品与生态
    "https://huggingface.co/blog/feed.xml",
    "https://www.microsoft.com/en-us/research/blog/feed/",
    "https://blog.google/rss/",
    "https://openai.com/blog/rss.xml",

    # 技术与行业资讯
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://www.technologyreview.com/feed/",

    # 投融资与商业动态
    "https://techcrunch.com/feed/",
    "https://www.theinformation.com/feed",  # 若不可用可删除
    "https://www.ft.com/technology?format=rss",

    # 咨询与研究机构
    "https://www.gartner.com/en/newsroom/rss",
    "https://www.mckinsey.com/featured-insights/rss",
    "https://www.bcg.com/rss",
    "https://www.bain.com/rss/",

    # 论文补充
    "https://arxiv.org/rss/cs.AI",
]


def _safe_parse_time(s: str):
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None


def fetch_recent_items(hours=RECENT_HOURS, max_items=MAX_ITEMS_TOTAL):
    now = datetime.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    cutoff = now - datetime.timedelta(hours=hours)

    items = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for e in getattr(feed, "entries", [])[:80]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()

            published = getattr(e, "published", "") or getattr(e, "updated", "") or getattr(e, "pubDate", "")
            published_dt = _safe_parse_time(published)

            if published_dt:
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=ZoneInfo("UTC"))
                published_dt_bj = published_dt.astimezone(ZoneInfo("Asia/Shanghai"))
            else:
                published_dt_bj = None

            summary = getattr(e, "summary", "") or getattr(e, "description", "")

            if not title or not link:
                continue

            # 48h 内过滤：如果拿不到时间就先保留，交给模型判断时效性并降低置信度
            if published_dt_bj and published_dt_bj < cutoff:
                continue

            items.append(
                {
                    "title": title,
                    "url": link,
                    "published": published_dt_bj.isoformat() if published_dt_bj else published,
                    "summary": " ".join(summary.replace("\n", " ").split())[:400],
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
            f"时间: {it['published']}\n"
            f"摘要: {it['summary']}\n"
        )
    return "\n".join(lines)


def call_deepseek_digest(input_text: str):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    schema_hint = {
        "date": "YYYY-MM-DD",
        "window_hours": 48,
        "top5": [
            {
                "title_cn": "中文标题",
                "one_liner_cn": "一句话总结（中文）",
                "why_it_matters_cn": "为什么重要（中文，偏商业与投融资视角）",
                "market_implication_cn": "资本市场含义（中文，预期差/催化剂/风险）",
                "product_implication_cn": "产品与应用含义（中文，落地场景/商业化）",
                "confidence": "high|medium|low",
                "urls": ["url"]
            }
        ],
        "investment_financing": [
            {
                "deal": "融资/并购/合作事件",
                "who": "公司/机构",
                "what": "发生了什么",
                "why": "对行业与资金的意义",
                "confidence": "high|medium|low",
                "urls": ["url"]
            }
        ],
        "company_product_updates": [
            {
                "company": "公司名",
                "product": "产品名或模块",
                "update": "更新内容",
                "who_cares": "谁会在意（开发者/企业/消费者）",
                "adoption_barriers": ["落地阻力1", "落地阻力2"],
                "confidence": "high|medium|low",
                "urls": ["url"]
            }
        ],
        "watchlist": [
            {
                "item_cn": "跟踪项",
                "metric_cn": "要盯的指标",
                "timeframe": "7d|14d|30d|this_week",
                "why_cn": "原因"
            }
        ],
        "sources": ["url"]
    }

    prompt = f"""
你是“AI 资讯日报编辑”，读者是关注公司动态、产品发布、投融资的商业人群。
请基于输入材料生成一份 48 小时 AI 日报。

硬性要求：
1) 输出必须为严格 JSON，结构参考 schema_hint。
2) 所有文字字段必须是中文输出。
3) 事实只能来自输入材料，不要编造。
4) 每条结论必须给出 urls（1-3 个）。
5) 同一事件多条来源合并为一条，避免重复。
6) 只保留最近 48 小时内的内容。若时间不明确，可以保留但 confidence 不能为 high。

schema_hint:
{json.dumps(schema_hint, ensure_ascii=False)}

输入材料：
{input_text}
""".strip()

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def render_text_cn(digest: dict):
    date = digest.get("date", "")
    top5 = digest.get("top5", [])
    inv = digest.get("investment_financing", [])
    updates = digest.get("company_product_updates", [])
    watchlist = digest.get("watchlist", [])
    sources = digest.get("sources", [])

    def fmt_urls(urls):
        return "；".join(urls[:3]) if urls else ""

    lines = []
    lines.append(f"AI 资讯日报（近48小时） | {date}")
    lines.append("")

    lines.append("一、今日要点 Top 5")
    if not top5:
        lines.append("暂无")
    else:
        for i, x in enumerate(top5, 1):
            lines.append(f"{i}. {x.get('title_cn','')}")
            lines.append(f"   摘要：{x.get('one_liner_cn','')}")
            lines.append(f"   为什么重要：{x.get('why_it_matters_cn','')}")
            lines.append(f"   资本市场：{x.get('market_implication_cn','')}")
            lines.append(f"   产品与应用：{x.get('product_implication_cn','')}")
            lines.append(f"   置信度：{x.get('confidence','')}")
            lines.append(f"   链接：{fmt_urls(x.get('urls', []))}")
            lines.append("")

    lines.append("二、投融资与商业合作")
    if not inv:
        lines.append("暂无")
    else:
        for i, x in enumerate(inv, 1):
            lines.append(f"{i}. {x.get('deal','')}")
            lines.append(f"   主体：{x.get('who','')}")
            lines.append(f"   发生了什么：{x.get('what','')}")
            lines.append(f"   意义：{x.get('why','')}")
            lines.append(f"   置信度：{x.get('confidence','')}")
            lines.append(f"   链接：{fmt_urls(x.get('urls', []))}")
            lines.append("")

    lines.append("三、公司动态与产品发布")
    if not updates:
        lines.append("暂无")
    else:
        for i, x in enumerate(updates, 1):
            lines.append(f"{i}. {x.get('company','')} | {x.get('product','')}")
            lines.append(f"   更新：{x.get('update','')}")
            lines.append(f"   谁会在意：{x.get('who_cares','')}")
            barriers = x.get("adoption_barriers", []) or []
            if barriers:
                lines.append(f"   落地阻力：{'；'.join(barriers[:4])}")
            lines.append(f"   置信度：{x.get('confidence','')}")
            lines.append(f"   链接：{fmt_urls(x.get('urls', []))}")
            lines.append("")

    lines.append("四、Watchlist（未来7-30天）")
    if not watchlist:
        lines.append("暂无")
    else:
        for i, x in enumerate(watchlist, 1):
            lines.append(f"{i}. {x.get('item_cn','')}")
            lines.append(f"   指标：{x.get('metric_cn','')}")
            lines.append(f"   时间：{x.get('timeframe','')}")
            lines.append(f"   原因：{x.get('why_cn','')}")
            lines.append("")

    lines.append("五、Sources")
    if not sources:
        lines.append("暂无")
    else:
        for u in sources[:20]:
            lines.append(f"- {u}")

    msg = "\n".join(lines)
    return msg[:18000]


def send_feishu_text(text: str):
    if not FEISHU_WEBHOOK:
        raise RuntimeError("Missing FEISHU_WEBHOOK")
    payload = {"msg_type": "text", "content": {"text": text}}
    r = requests.post(FEISHU_WEBHOOK, json=payload, timeout=30)
    r.raise_for_status()


def main():
    now_bj = datetime.datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    items = fetch_recent_items(hours=RECENT_HOURS, max_items=MAX_ITEMS_TOTAL)

    if not items:
        send_feishu_text(f"AI 资讯日报（近48小时） | {now_bj.date().isoformat()}\n\n过去 48 小时未抓取到条目。")
        return

    input_text = build_input_text(items)
    digest = call_deepseek_digest(input_text)

    if not digest.get("date"):
        digest["date"] = now_bj.date().isoformat()

    msg = render_text_cn(digest)
    send_feishu_text(msg)


if __name__ == "__main__":
    main()
