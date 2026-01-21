import os
import sys
import json
import re
import datetime
from zoneinfo import ZoneInfo

import requests
import feedparser
from dateutil import parser as dtparser


# =========================
# 基础配置
# =========================
TZ = ZoneInfo("Asia/Shanghai")

RECENT_HOURS = 48

# 条目上限建议不要太大，避免模型输出过长导致 JSON 更容易损坏
MAX_ITEMS_TOTAL = 55

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

FEISHU_WEBHOOK_MORNING = os.getenv("FEISHU_WEBHOOK_MORNING", "").strip()
FEISHU_WEBHOOK_AFTERNOON = os.getenv("FEISHU_WEBHOOK_AFTERNOON", "").strip()

# RSSHub 基地址
# 建议你在 GitHub Secrets 里配置 RSSHUB_BASE
# 例如：https://rsshub.app
RSSHUB_BASE = os.getenv("RSSHUB_BASE", "https://rsshub.app").strip().rstrip("/")


def now_bj() -> datetime.datetime:
    return datetime.datetime.now(tz=TZ)


def safe_parse_time(s: str):
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None


# =========================
# JSON 解析兜底工具
# =========================
def extract_json_object(text: str) -> str:
    """
    从模型输出中尽可能提取第一个 JSON 对象 {...}
    防止模型在 JSON 前后加额外说明。
    """
    if not text:
        return ""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_json_text(text: str) -> str:
    """
    尝试做轻量清洗，提升解析成功率
    """
    if not text:
        return text

    t = text.strip()

    # 去掉 ```json ``` 包裹
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    return t.strip()


# =========================
# 信息源配置
# =========================
RSS_FEEDS_EN = [
    # 官方与产品动态
    "https://openai.com/blog/rss.xml",
    "https://blog.google/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://www.microsoft.com/en-us/research/blog/feed/",

    # 商业与科技媒体
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.technologyreview.com/feed/",

    # 论文补充
    "https://arxiv.org/rss/cs.AI",
]

# 中文主流科技媒体 + 投融资 + 官方渠道
# 说明：很多中文站点没有稳定 RSS，使用 RSSHub 转换
# 若某个 route 失效，Actions log 会显示抓取条目变少，你把 route 发我，我帮你换
RSS_FEEDS_CN = [
    # A) 科技媒体（快讯密度高）
    f"{RSSHUB_BASE}/36kr/newsflashes",
    f"{RSSHUB_BASE}/huxiu/article",
    f"{RSSHUB_BASE}/geekpark/news",
    f"{RSSHUB_BASE}/qbitai/category",
    f"{RSSHUB_BASE}/jiqizhixin/news",

    # B) 创投与商业资讯（投融资更集中）
    f"{RSSHUB_BASE}/chinaventure/news",
    f"{RSSHUB_BASE}/pedaily/news",
    f"{RSSHUB_BASE}/itjuzi/invest",

    # C) 大厂与平台官方（产品发布更权威）
    f"{RSSHUB_BASE}/aliyun/notice",
    f"{RSSHUB_BASE}/tencentcloud/notice",
    f"{RSSHUB_BASE}/huaweicloud/notice",
    f"{RSSHUB_BASE}/volcengine/notice",
]

RSS_FEEDS = RSS_FEEDS_EN + RSS_FEEDS_CN


# =========================
# 抓取与过滤（严格 48h）
# =========================
def fetch_recent_items(hours=RECENT_HOURS, max_items=MAX_ITEMS_TOTAL):
    cutoff = now_bj() - datetime.timedelta(hours=hours)

    items = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        entries = getattr(feed, "entries", []) or []
        feed_title = getattr(getattr(feed, "feed", None), "title", "") or feed_url

        for e in entries[:150]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            if not title or not link:
                continue

            published = getattr(e, "published", "") or getattr(e, "updated", "") or getattr(e, "pubDate", "")
            published_dt = safe_parse_time(published)
            if not published_dt:
                # 严格时效：没有时间直接丢弃
                continue

            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=ZoneInfo("UTC"))
            published_dt_bj = published_dt.astimezone(TZ)

            if published_dt_bj < cutoff:
                continue

            summary = getattr(e, "summary", "") or getattr(e, "description", "")
            summary = " ".join(summary.replace("\n", " ").split())[:600]

            items.append(
                {
                    "source": feed_title,
                    "title": title,
                    "url": link,
                    "published": published_dt_bj.isoformat(),
                    "summary": summary,
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

    # 按时间倒序
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


# =========================
# DeepSeek 调用与 JSON 修复兜底
# =========================
def deepseek_chat_json(prompt: str, timeout=120):
    """
    返回 dict，确保可解析 JSON
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()

    content = r.json()["choices"][0]["message"]["content"]
    content = normalize_json_text(content)

    # 1) 直接解
