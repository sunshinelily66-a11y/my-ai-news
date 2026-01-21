import os
import datetime
from zoneinfo import ZoneInfo

import requests
import feedparser
from dateutil import parser as dtparser


TZ = ZoneInfo("Asia/Shanghai")
RECENT_HOURS = 48

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
FEISHU_WEBHOOK_MORNING = os.getenv("FEISHU_WEBHOOK_MORNING", "").strip()

RSSHUB_BASE = os.getenv("RSSHUB_BASE", "https://rsshub.app").strip().rstrip("/")

MAX_ITEMS_TOTAL = 40
TOP_N = 8

RSS_FEEDS_EN = [
    "https://openai.com/blog/rss.xml",
    "https://blog.google/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://www.microsoft.com/en-us/research/blog/feed/",
    "https://techcrunch.com/feed/",
    "https://www.technologyreview.com/feed/",
]

RSS_FEEDS_CN = [
    f"{RSSHUB_BASE}/36kr/newsflashes",
    f"{RSSHUB_BASE}/huxiu/article",
    f"{RSSHUB_BASE}/geekpark/news",
    f"{RSSHUB_BASE}/qbitai/category",
    f"{RSSHUB_BASE}/jiqizhixin/news",
    f"{RSSHUB_BASE}/chinaventure/news",
    f"{RSSHUB_BASE}/pedaily/news",
    f"{RSSHUB_BASE}/itjuzi/invest",
    f"{RSSHUB_BASE}/aliyun/notice",
    f"{RSSHUB_BASE}/tencentcloud/notice",
    f"{RSSHUB_BASE}/huaweicloud/notice",
    f"{RSSHUB_BASE}/volcengine/notice",
]

RSS_FEEDS = RSS_FEEDS_EN + RSS_FEEDS_CN


def now_bj() -> datetime.datetime:
    return datetime.datetime.now(tz=TZ)


def safe_parse_time(s: str):
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None


def fetch_recent_items(hours: int = RECENT_HOURS, max_items: int = MAX_ITEMS_TOTAL):
    cutoff = now_bj() - datetime.timedelta(hours=hours)

    items = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        entries = getattr(feed, "entries", []) or []
        source = getattr(getattr(feed, "feed", None), "title", "") or feed_url

        for e in entries[:150]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            if not title or not link:
                continue

            published = getattr(e, "published", "") or getattr(e, "updated", "") or getattr(e, "pubDate", "")
            dt = safe_parse_time(published)
            if not dt:
                continue

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            dt_bj = dt.astimezone(TZ)
            if dt_bj < cutoff:
                continue

            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            summary = " ".join(summary.replace("\n", " ").split())
            if len(summary) > 260:
                summary = summary[:260] + "…"

            items.append(
                {
                    "source": source,
                    "title": title,
                    "url": link,
                    "published": dt_bj.isoformat(),
                    "summary": summary,
                }
            )

    seen = set()
    deduped = []
    for it in items:
        key = it["url"].split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    deduped.sort(key=lambda x: x["published"], reverse=True)
    return deduped[:max_items]


def build_brief_input(items):
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. 标题：{it['title']}\n"
            f"来源：{it['source']}\n"
            f"时间：{it['published']}\n"
            f"链接：{it['url']}\n"
            f"摘要：{it['summary']}\n"
        )
    text = "\n".join(lines)
    if len(text) > 12000:
        text = text[:12000] + "\n\n内容过长已截断"
    return text


def call_deepseek_text(prompt: str, max_tokens: int = 1400, timeout: int = 90) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def render_fallback(items):
    date = now_bj().date().isoformat()
    lines = [f"AI 晨报（近48小时） | {date}", "", "模型分析失败，已降级为资讯列表。", ""]
    for i, it in enumerate(items[:20], 1):
        lines.append(f"{i}. {it['title']}")
        lines.append(f"来源：{it['source']}")
        lines.append(f"时间：{it['published']}")
        lines.append(f"链接：{it['url']}")
        lines.append("")
    text = "\n".join(lines)
    return text[:18000]


def render_report(analysis_text: str, top_items):
    date = now_bj().date().isoformat()
    lines = [f"AI 晨报（近48小时） | {date}", "", analysis_text.strip(), "", "参考链接", ""]
    for i, it in enumerate(top_items, 1):
        lines.append(f"{i}. {it['title']}")
        lines.append(f"{it['url']}")
        lines.append("")
    text = "\n".join(lines)
    return text[:18000]


def send_feishu(text: str):
    if not FEISHU_WEBHOOK_MORNING:
        raise RuntimeError("Missing FEISHU_WEBHOOK_MORNING")

    payload = {"msg_type": "text", "content": {"text": text}}
    r = requests.post(FEISHU_WEBHOOK_MORNING, json=payload, timeout=30)
    r.raise_for_status()


def main():
    items = fetch_recent_items()
    if not items:
        date = now_bj().date().isoformat()
        send_feishu(f"AI 晨报（近48小时） | {date}\n\n过去48小时未抓取到可用条目。")
        return

    top_items = items[:TOP_N]
    input_text = build_brief_input(top_items)

    prompt = f"""
你是面向商业读者的 AI 资讯晨报编辑。请基于输入材料输出中文晨报，窗口为最近48小时。
输出要求：
1）先给一句总览，最多40字
2）再给重点事件清单，最多6条，每条格式固定为：
事件：一句话
影响：用商业和投融资视角写一句话
3）再给一个 Watchlist，写4条，每条包含：指标，时间范围，原因
4）不需要输出JSON，不要编号以外的额外格式，不要输出免责声明
5）内容必须来自输入材料，不要编造公司、融资金额、产品功能

输入材料：
{input_text}
""".strip()

    try:
        analysis = call_deepseek_text(prompt, max_tokens=1400)
        msg = render_report(analysis, top_items)
        send_feishu(msg)
    except Exception as e:
        print(f"DeepSeek call failed: {e}")
        msg = render_fallback(items)
        send_feishu(msg)


if __name__ == "__main__":
    main()
