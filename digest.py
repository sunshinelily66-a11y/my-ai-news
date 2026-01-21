import os
import sys
import json
import datetime
from zoneinfo import ZoneInfo

import requests
import feedparser
from dateutil import parser as dtparser


# ===== 基础配置 =====
RECENT_HOURS = 48
MAX_ITEMS_TOTAL = 80  # 中文源加多后，条目会变多，留足空间给模型筛选
TZ = ZoneInfo("Asia/Shanghai")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
FEISHU_WEBHOOK_MORNING = os.getenv("FEISHU_WEBHOOK_MORNING", "").strip()
FEISHU_WEBHOOK_AFTERNOON = os.getenv("FEISHU_WEBHOOK_AFTERNOON", "").strip()

# RSSHub 基地址（你需要改成可用的实例）
# 示例：RSSHUB_BASE="https://rsshub.app"
RSSHUB_BASE = os.getenv("RSSHUB_BASE", "https://rsshub.app").strip().rstrip("/")


def _safe_parse_time(s: str):
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None


def _now_bj():
    return datetime.datetime.now(tz=TZ)


# ===== 信息源 =====
RSS_FEEDS_EN = [
    "https://openai.com/blog/rss.xml",
    "https://blog.google/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://www.microsoft.com/en-us/research/blog/feed/",
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.technologyreview.com/feed/",
    "https://arxiv.org/rss/cs.AI",
]

# 中文媒体与官方渠道（建议走 RSSHub）
# 这些 route 可能会因 RSSHub 版本变化而不同
# 你先跑一遍，哪个报错我再帮你替换为可用 route
RSS_FEEDS_CN = [
    # A) 科技媒体快讯
    f"{RSSHUB_BASE}/36kr/newsflashes",
    f"{RSSHUB_BASE}/huxiu/article",
    f"{RSSHUB_BASE}/geekpark/news",
    f"{RSSHUB_BASE}/qbitai/category",
    f"{RSSHUB_BASE}/jiqizhixin/news",

    # B) 创投与商业资讯
    f"{RSSHUB_BASE}/chinaventure/news",
    f"{RSSHUB_BASE}/pedaily/news",
    f"{RSSHUB_BASE}/itjuzi/invest",

    # C) 大厂与平台官方（优先“发布/更新/公告”）
    f"{RSSHUB_BASE}/aliyun/notice",
    f"{RSSHUB_BASE}/tencentcloud/notice",
    f"{RSSHUB_BASE}/huaweicloud/notice",
    f"{RSSHUB_BASE}/volcengine/notice",
]

RSS_FEEDS = RSS_FEEDS_EN + RSS_FEEDS_CN


def fetch_recent_items(hours=RECENT_HOURS, max_items=MAX_ITEMS_TOTAL):
    now = _now_bj()
    cutoff = now - datetime.timedelta(hours=hours)

    items = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for e in getattr(feed, "entries", [])[:120]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()

            published = getattr(e, "published", "") or getattr(e, "updated", "") or getattr(e, "pubDate", "")
            published_dt = _safe_parse_time(published)

            published_dt_bj = None
            if published_dt:
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=ZoneInfo("UTC"))
                published_dt_bj = published_dt.astimezone(TZ)

            summary = getattr(e, "summary", "") or getattr(e, "description", "")
            source = getattr(feed.feed, "title", "") or feed_url

            if not title or not link:
                continue

            # 严格 48h：如果拿不到时间，直接丢弃，保证时效性
            if not published_dt_bj:
                continue
            if published_dt_bj < cutoff:
                continue

            items.append(
                {
                    "title": title,
                    "url": link,
                    "published": published_dt_bj.isoformat(),
                    "summary": " ".join(summary.replace("\n", " ").split())[:500],
                    "source": source,
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
    deduped.sort(key=lambda x: x["published"], reverse=True)
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


def call_deepseek(mode: str, input_text: str):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    if mode == "morning":
        style = "晨报研报型"
        requirements = """
输出目标：更像投研晨报，强调“公司动态、产品发布、投融资”，少而精。
约束：
- Top 8 以内
- 每条都要写：事件是什么、为什么重要、对资本市场的含义、对产品与应用的含义、潜在催化剂、主要风险
- 结尾给 Watchlist（5条）
"""
        schema_hint = {
            "date": "YYYY-MM-DD",
            "window_hours": 48,
            "style": "晨报研报型",
            "top": [
                {
                    "title_cn": "中文标题",
                    "event_cn": "发生了什么",
                    "why_cn": "为什么重要",
                    "market_cn": "资本市场含义（预期差/催化剂/风险）",
                    "product_cn": "产品与应用含义（落地/商业化/竞品）",
                    "catalysts_cn": ["催化剂1", "催化剂2"],
                    "risks_cn": ["风险1", "风险2"],
                    "confidence": "high|medium|low",
                    "urls": ["url"]
                }
            ],
            "financing": [
                {
                    "deal_cn": "融资/并购/合作",
                    "players_cn": "参与方",
                    "take_cn": "一句话判断",
                    "confidence": "high|medium|low",
                    "urls": ["url"]
                }
            ],
            "watchlist": [
                {
                    "item_cn": "跟踪项",
                    "metric_cn": "指标",
                    "timeframe": "7d|14d|30d|this_week",
                    "why_cn": "原因"
                }
            ],
            "sources": ["url"]
        }
    else:
        style = "资讯快报型"
        requirements = """
输出目标：更像资讯快讯流，覆盖更多信息点，快速扫一遍就能掌握。
约束：
- Top 15 以内
- 每条只写：一句话 + 标签 + 链接
- 单独列出“投融资/合作”快讯区块
"""
        schema_hint = {
            "date": "YYYY-MM-DD",
            "window_hours": 48,
            "style": "资讯快报型",
            "briefs": [
                {
                    "title_cn": "中文标题",
                    "one_liner_cn": "一句话快报",
                    "tags": ["公司动态", "产品发布", "投融资", "监管", "开源"],
                    "confidence": "high|medium|low",
                    "urls": ["url"]
                }
            ],
            "financing_briefs": [
                {
                    "one_liner_cn": "一句话投融资/合作快讯",
                    "confidence": "high|medium|low",
                    "urls": ["url"]
                }
            ],
            "sources": ["url"]
        }

    prompt = f"""
你是 AI 资讯编辑。请基于输入材料生成【{style}】日报，窗口为最近 48 小时。
硬性要求：
1) 全部中文输出。
2) 事实只能来自输入材料，不要编造。
3) 同一事件多条来源合并，避免重复。
4) 每条必须带 urls（1-3个）。
5) 输出必须是严格 JSON，结构参考 schema_hint。

{requirements}

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

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def render_text(mode: str, digest: dict):
    date = digest.get("date", _now_bj().date().isoformat())
    style = digest.get("style", "")

    def join_urls(urls):
        return "；".join((urls or [])[:3])

    lines = []
    if mode == "morning":
        lines.append(f"AI 晨报（近48小时） | {date}")
        lines.append(f"风格：{style}")
        lines.append("")
        top = digest.get("top", [])
        lines.append("一、重点事件")
        if not top:
            lines.append("暂无")
        else:
            for i, x in enumerate(top[:8], 1):
                lines.append(f"{i}. {x.get('title_cn','')}")
                lines.append(f"   事件：{x.get('event_cn','')}")
                lines.append(f"   重要性：{x.get('why_cn','')}")
                lines.append(f"   资本市场：{x.get('market_cn','')}")
                lines.append(f"   产品与应用：{x.get('product_cn','')}")
                catalysts = x.get("catalysts_cn", []) or []
                risks = x.get("risks_cn", []) or []
                if catalysts:
                    lines.append(f"   催化剂：{'；'.join(catalysts[:3])}")
                if risks:
                    lines.append(f"   风险：{'；'.join(risks[:3])}")
                lines.append(f"   置信度：{x.get('confidence','')}")
                lines.append(f"   链接：{join_urls(x.get('urls', []))}")
                lines.append("")

        lines.append("二、投融资与合作")
        fin = digest.get("financing", [])
        if not fin:
            lines.append("暂无")
        else:
            for i, x in enumerate(fin[:8], 1):
                lines.append(f"{i}. {x.get('deal_cn','')}")
                lines.append(f"   参与方：{x.get('players_cn','')}")
                lines.append(f"   判断：{x.get('take_cn','')}")
                lines.append(f"   置信度：{x.get('confidence','')}")
                lines.append(f"   链接：{join_urls(x.get('urls', []))}")
                lines.append("")

        lines.append("三、Watchlist")
        wl = digest.get("watchlist", [])
        if not wl:
            lines.append("暂无")
        else:
            for i, x in enumerate(wl[:6], 1):
                lines.append(f"{i}. {x.get('item_cn','')}")
                lines.append(f"   指标：{x.get('metric_cn','')}")
                lines.append(f"   时间：{x.get('timeframe','')}")
                lines.append(f"   原因：{x.get('why_cn','')}")
                lines.append("")
    else:
        lines.append(f"AI 快报（近48小时） | {date}")
        lines.append(f"风格：{style}")
        lines.append("")
        briefs = digest.get("briefs", [])
        lines.append("一、快讯")
        if not briefs:
            lines.append("暂无")
        else:
            for i, x in enumerate(briefs[:15], 1):
                tags = x.get("tags", []) or []
                tag_str = "、".join(tags[:5])
                lines.append(f"{i}. {x.get('title_cn','')}")
                lines.append(f"   {x.get('one_liner_cn','')}")
                if tag_str:
                    lines.append(f"   标签：{tag_str}")
                lines.append(f"   置信度：{x.get('confidence','')}")
                lines.append(f"   链接：{join_urls(x.get('urls', []))}")
                lines.append("")

        lines.append("二、投融资/合作快讯")
        fbriefs = digest.get("financing_briefs", [])
        if not fbriefs:
            lines.append("暂无")
        else:
            for i, x in enumerate(fbriefs[:10], 1):
                lines.append(f"{i}. {x.get('one_liner_cn','')}")
                lines.append(f"   置信度：{x.get('confidence','')}")
                lines.append(f"   链接：{join_urls(x.get('urls', []))}")
                lines.append("")

    # Sources 可选，不在群里刷屏
    msg = "\n".join(lines)
    return msg[:18000]


def send_feishu(webhook: str, text: str):
    if not webhook:
        raise RuntimeError("Missing Feishu webhook for this mode")
    payload = {"msg_type": "text", "content": {"text": text}}
    r = requests.post(webhook, json=payload, timeout=30)
    r.raise_for_status()


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "morning").strip().lower()
    if mode not in ["morning", "afternoon"]:
        raise ValueError("mode must be morning or afternoon")

    items = fetch_recent_items(hours=RECENT_HOURS, max_items=MAX_ITEMS_TOTAL)
    if not items:
        fallback = f"AI {'晨报' if mode=='morning' else '快报'}（近48小时） | {_now_bj().date().isoformat()}\n\n过去48小时未抓取到可用条目。"
        webhook = FEISHU_WEBHOOK_MORNING if mode == "morning" else FEISHU_WEBHOOK_AFTERNOON
        send_feishu(webhook, fallback)
        return

    input_text = build_input_text(items)
    digest = call_deepseek(mode=mode, input_text=input_text)
    text = render_text(mode=mode, digest=digest)

    webhook = FEISHU_WEBHOOK_MORNING if mode == "morning" else FEISHU_WEBHOOK_AFTERNOON
    send_feishu(webhook, text)


if __name__ == "__main__":
    main()
