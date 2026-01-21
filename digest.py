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
MAX_ITEMS_TOTAL = 35

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

    # 1) 直接解析
    parsed = try_parse_json(content)
    if parsed is not None:
        return parsed

    # 2) 抽取 {...} 解析
    extracted = extract_json_object(content)
    extracted = normalize_json_text(extracted)
    parsed = try_parse_json(extracted)
    if parsed is not None:
        return parsed

    # 3) 自动修复再解析
    print("DeepSeek raw output (first 900 chars):")
    print(content[:900])

    repair_prompt = f"""
你刚刚输出的内容不是严格 JSON，导致解析失败。
请将下面内容修复为严格 JSON，并且只输出 JSON 本体，不要输出任何解释文字。

待修复内容：
{content}
""".strip()

    payload_repair = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": repair_prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    r2 = requests.post(url, headers=headers, data=json.dumps(payload_repair), timeout=timeout)
    r2.raise_for_status()
    content2 = r2.json()["choices"][0]["message"]["content"]
    content2 = normalize_json_text(content2)

    parsed = try_parse_json(content2)
    if parsed is not None:
        return parsed

    extracted2 = extract_json_object(content2)
    extracted2 = normalize_json_text(extracted2)
    parsed = try_parse_json(extracted2)
    if parsed is not None:
        return parsed

    print("DeepSeek repaired output (first 900 chars):")
    print(content2[:900])
    raise RuntimeError("DeepSeek output is not valid JSON even after repair.")


def build_prompt(mode: str, input_text: str) -> str:
    """
    mode:
      - morning: 晨报研报型（少而精，分析更深）
      - afternoon: 资讯快报型（快而密，覆盖更多）
    """
    if mode == "morning":
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

        requirements = """
输出目标：更像投研晨报，强调公司动态、产品发布、投融资，少而精。
约束：
- top 最多 8 条
- 每条都写：事件、重要性、资本市场含义、产品与应用含义、催化剂、风险
- financing 最多 8 条
- watchlist 给 5 条
"""

    else:
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

        requirements = """
输出目标：更像资讯快讯流，覆盖更多信息点，快速扫一遍就能掌握。
约束：
- briefs 最多 15 条
- 每条一句话，附标签与链接
- 单独列出融资/合作快讯区块，最多 10 条
"""

    prompt = f"""
你是 AI 资讯编辑。请基于输入材料生成日报，窗口为最近 48 小时。
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

    return prompt


# =========================
# 渲染为飞书文本
# =========================
def render_text(mode: str, digest: dict) -> str:
    date = digest.get("date") or now_bj().date().isoformat()
    style = digest.get("style", "")

    def join_urls(urls):
        return "；".join((urls or [])[:3])

    lines = []

    if mode == "morning":
        lines.append(f"AI 晨报（近48小时） | {date}")
        if style:
            lines.append(f"风格：{style}")
        lines.append("")

        top = digest.get("top", []) or []
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

        fin = digest.get("financing", []) or []
        lines.append("二、投融资与合作")
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

        wl = digest.get("watchlist", []) or []
        lines.append("三、Watchlist")
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
        if style:
            lines.append(f"风格：{style}")
        lines.append("")

        briefs = digest.get("briefs", []) or []
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

        fbriefs = digest.get("financing_briefs", []) or []
        lines.append("二、投融资/合作快讯")
        if not fbriefs:
            lines.append("暂无")
        else:
            for i, x in enumerate(fbriefs[:10], 1):
                lines.append(f"{i}. {x.get('one_liner_cn','')}")
                lines.append(f"   置信度：{x.get('confidence','')}")
                lines.append(f"   链接：{join_urls(x.get('urls', []))}")
                lines.append("")

    msg = "\n".join(lines)
    return msg[:18000]


# =========================
# 飞书推送
# =========================
def send_feishu(webhook: str, text: str):
    if not webhook:
        raise RuntimeError("Missing Feishu webhook")

    payload = {"msg_type": "text", "content": {"text": text}}
    r = requests.post(webhook, json=payload, timeout=30)
    r.raise_for_status()


# =========================
# 主入口
# =========================
def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "morning").strip().lower()
    if mode not in ["morning", "afternoon"]:
        raise ValueError("mode must be morning or afternoon")

    webhook = FEISHU_WEBHOOK_MORNING if mode == "morning" else FEISHU_WEBHOOK_AFTERNOON
    label = "晨报" if mode == "morning" else "快报"

    items = fetch_recent_items(hours=RECENT_HOURS, max_items=MAX_ITEMS_TOTAL)

    if not items:
        fallback = f"AI {label}（近48小时） | {now_bj().date().isoformat()}\n\n过去48小时未抓取到可用条目。"
        send_feishu(webhook, fallback)
        return

    input_text = build_input_text(items)
    def truncate_text(s: str, max_chars: int) -> str:
    if not s:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n（内容过长已截断）"
    prompt = build_prompt(mode=mode, input_text=input_text)

    def deepseek_chat_json(prompt: str, timeout=120, max_tokens=2200):
    """
    返回 dict，确保可解析 JSON
    增加 max_tokens，避免输出被截断导致 JSON 断裂
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    # prompt 太长会显著提高截断概率，先做硬截断
    prompt = truncate_text(prompt, max_chars=14000)

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()

    content = r.json()["choices"][0]["message"]["content"]
    content = normalize_json_text(content)

    # 1) 直接解析
    parsed = try_parse_json(content)
    if parsed is not None:
        return parsed

    # 2) 抽取 {...} 解析
    extracted = extract_json_object(content)
    extracted = normalize_json_text(extracted)
    parsed = try_parse_json(extracted)
    if parsed is not None:
        return parsed

    # 3) 自动修复再解析
    print("DeepSeek raw output (first 900 chars):")
    print(content[:900])

    repair_prompt = f"""
你刚刚输出的内容不是严格 JSON，导致解析失败。
请将下面内容修复为严格 JSON，并且只输出 JSON 本体，不要输出任何解释文字。

待修复内容：
{content}
""".strip()

    payload_repair = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": repair_prompt}],
        "temperature": 0,
        "max_tokens": 2200,
        "response_format": {"type": "json_object"},
    }

    r2 = requests.post(url, headers=headers, data=json.dumps(payload_repair), timeout=timeout)
    r2.raise_for_status()
    content2 = r2.json()["choices"][0]["message"]["content"]
    content2 = normalize_json_text(content2)

    parsed = try_parse_json(content2)
    if parsed is not None:
        return parsed

    extracted2 = extract_json_object(content2)
    extracted2 = normalize_json_text(extracted2)
    parsed = try_parse_json(extracted2)
    if parsed is not None:
        return parsed

    print("DeepSeek repaired output (first 900 chars):")
    print(content2[:900])

    raise RuntimeError("DeepSeek output is not valid JSON even after repair.")

    text = render_text(mode=mode, digest=digest)

    send_feishu(webhook, text)


if __name__ == "__main__":
    main()
