"""Source normalisation, scoring and rendering.

Every collected page passes through here before a model reads it: URL identity,
authority-derived confidence, excerpting, and the numbered index used in prompts.
"""

import re
from collections.abc import Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunsplit

from agents.research_team.contracts import Source

_LANG_SEGMENT = re.compile(r"/(?:zh-cn|zh-tw|cn|tw|hk|en|zh)(?=/|$)", re.IGNORECASE)
_MARKDOWN_LINK_ONLY = re.compile(r"^\[([^\]]+)\]\((.+)\)$")
_IMAGE_ONLY = re.compile(r"^!\[[^\]]*\]\([^)]+\)$")
_BREADCRUMB_LINE = re.compile(r"^\s*\S+(?:\s*>\s*\S+)+\s*$")
_FETCH_NOISE_LABELS = frozenset(
    {
        "首页",
        "登录",
        "注册",
        "分享",
        "打印",
        "纠错",
        "我要纠错",
        "相关稿件",
        "推荐阅读",
        "相关阅读",
    }
)
_FETCH_NOISE_LINE = re.compile(
    r"^(?:"
    r"责任编辑\s*[:：].*"
    r"|来源网站责任编辑\s*[:：].*"
    r"|.*纠错.*责任编辑.*"
    r"|(?:\[?纠错\]?|【纠错】)(?:\s+.*)?"
    r"|(?:字体|字号)\s*[:：].*"
    r"|Copyright\s+©?\s*\d{4}.*"
    r"|版权所有.*"
    r"|制作单位\s*[:：].*版权所有.*"
    r")$",
    re.IGNORECASE,
)
_STANDALONE_URL = re.compile(r"^https?://\S+$", re.IGNORECASE)

_AUTHORITY_BASE = {4: 0.90, 3: 0.80, 2: 0.65, 1: 0.50, 0: 0.40}
# Search does not always report authority. The unknown case sits between the lowest two
# known levels rather than borrowing one: a missing score must not outrank a
# known-good source, and must not sink to the bottom either.
_AUTHORITY_UNKNOWN_BASE = 0.45

# How much of a source the analyst and writer actually read. Fetch stores up to
# MAX_CONTENT_CHARS, so a lower cap silently discards paid-for text *and* hides the
# discard from the model, which then reads "absent" where the truth is "cut off".
EXCERPT_CHARS = 1500

# Total excerpt text one prompt may carry. Spreading it over the window keeps a wider
# window at the same cost as a narrow one: 12 sources read 1500 characters each, 20
# sources read 900 each. Breadth is what coverage judgements need; depth is what a
# single-fact extraction needs, and the marker in `excerpt` tells the model which one
# it is looking at.
EXCERPT_BUDGET_CHARS = 18_000


def excerpt_limit_for(source_count: int) -> int:
    """Per-source excerpt cap under a fixed total budget."""
    if source_count <= 0:
        return EXCERPT_CHARS
    return max(1, min(EXCERPT_CHARS, EXCERPT_BUDGET_CHARS // source_count))


def clean_fetched_markdown(text: str) -> str:
    """Remove high-confidence webpage noise without summarizing the source."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
    lines: list[str] = []
    seen: set[str] = set()

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue

        if (
            _IMAGE_ONLY.fullmatch(line)
            or _BREADCRUMB_LINE.fullmatch(line)
            or _FETCH_NOISE_LINE.fullmatch(line)
            or _STANDALONE_URL.fullmatch(line)
        ):
            continue
        link = _MARKDOWN_LINK_ONLY.fullmatch(line)
        if link:
            label, target = link.groups()
            if label.strip() in _FETCH_NOISE_LABELS or target.startswith("javascript:"):
                continue

        # Table header/separator rows legitimately repeat across tables on one page;
        # deduping them would leave the second table as orphan rows.
        if not line.startswith("|"):
            if line in seen:
                continue
            seen.add(line)
        lines.append(line)

    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def url_key(url: str) -> str:
    """Normalise a URL so language and mobile variants of the same article collapse.

    DeepSeek Search happily returns /cn/ and /tw/ copies of the same page, plus
    m./www. host variants; those differ only by URL, so a naive URL comparison
    lets one article in twice and makes a single source look like two independent
    ones during cross-validation. The identity-bearing query is kept (via
    clean_url) so two different articles that share a path but differ by ?id=
    stay distinct.
    """
    parsed = urlparse(clean_url(url))
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    path = _LANG_SEGMENT.sub("", parsed.path.rstrip("/"))
    key = f"{host}{path}"
    return f"{key}?{parsed.query}" if parsed.query else key


def confidence_from_authority(authority: int, rerank_score: float) -> float:
    base = _AUTHORITY_BASE.get(authority, _AUTHORITY_UNKNOWN_BASE)
    if rerank_score >= 2.5:
        base += 0.10
    elif rerank_score >= 1.5:
        base += 0.05
    return round(min(base, 1.0), 2)


# Params that mark a share/channel, not the article itself. Dropping them lets the
# same article shared via different channels collapse; identity params (?id=) stay.
_TRACKING_PARAMS = frozenset(
    {
        "spm",
        "from",
        "ref",
        "referrer",
        "fbclid",
        "gclid",
        "scm",
        "share_token",
        "share_source",
        "vd_source",
    }
)


def clean_url(url: str) -> str:
    """Drop fragments and tracking params; keep identity-bearing query, sorted.

    Search appends anchors like `#1` and share tags like `?utm_source=`; those do
    not identify the article. But some sites put the article id in the query
    (`?id=1001`), so clearing the whole query would merge two different articles
    and fetch the wrong page. Sorting makes parameter order irrelevant.
    """
    parsed = urlparse(url.strip())
    pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS and not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(pairs))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def clean_title(title: str) -> str:
    """Collapse `Headline - Headline` titles that some news sites return."""
    head, separator, tail = title.strip().partition(" - ")
    if separator and head and tail.strip().startswith(head[:15]):
        return head.strip()
    return title.strip()


def dedupe_sources(sources: Sequence[Source] | None) -> list[Source]:
    """Keep the first occurrence of each page, preserving collection order.

    A later round may deep-read a URL the first round only skimmed, appending a
    second copy that carries the fetched body. First-wins alone would discard that
    paid-for content, so the kept copy is upgraded with the deeper material instead
    of mutated in place (state holds the originals).
    """
    seen: dict[str, Source] = {}
    order: list[str] = []
    for source in sources or []:
        key = url_key(source.url)
        if not key:
            continue
        kept = seen.get(key)
        if kept is None:
            seen[key] = source
            order.append(key)
            continue
        update: dict[str, str] = {}
        if len(source.content) > len(kept.content):
            update["content"] = source.content
        if len(source.summary) > len(kept.summary):
            update["summary"] = source.summary
        if update:
            seen[key] = kept.model_copy(update=update)
    return [seen[key] for key in order]


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Cut an excerpt and say so.

    A silent cut teaches the model that whatever it cannot see does not exist,
    which corrupts the "information gap" section it is asked to write. The marker
    is therefore part of the contract, not decoration.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…（节选，全文共 {len(text)} 字，此处为前 {limit} 字）"


def numbered_sources(
    sources: Sequence[Source],
    include_excerpt: bool = True,
    excerpt_chars: int | None = None,
) -> str:
    blocks = []
    limit = excerpt_chars or EXCERPT_CHARS
    for index, source in enumerate(sources, start=1):
        lines = [
            f"[{index}] {source.title or source.url}",
            f"    URL: {source.url}",
            f"    站点: {source.site_name or '未知'} | 发布: {source.published_at or '未知'}",
            f"    来源评分: {source.confidence:.2f}"
            f"（authority {source.authority}, rerank {source.rerank_score:.2f}）",
        ]
        # Skipping the line entirely, rather than printing a placeholder, keeps the
        # index-only form from spending tokens on "excerpt omitted" once per source.
        if include_excerpt:
            body = excerpt(source.content or source.summary or source.snippet, limit)
            lines.append(f"    摘录: {body}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def source_appendix(sources: Sequence[Source]) -> str:
    if not sources:
        return ""
    lines = ["", "---", "", "### 参考来源", ""]
    for index, source in enumerate(sources, start=1):
        title = source.title or source.url
        published = f"，{source.published_at[:10]}" if source.published_at else ""
        lines.append(
            f"[{index}] [{title}]({source.url}) — 来源评分 {source.confidence:.2f}"
            f"（{source.site_name or '未知来源'}{published}）"
        )
    return "\n".join(lines)


# Two digits is the whole citation space here (the ledger stays far below 100), and it
# keeps a year such as `[2024]` out of the rewrite. Dropping those would damage the
# prose instead of fixing a citation. One pattern, so clamping and remapping can never
# disagree about what counts as an anchor.
_ANCHOR = re.compile(r"\[(\d{1,2})\]")


def clamp_anchors(text: str, source_count: int) -> str:
    """Drop citation anchors that do not resolve to a recorded source."""

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= source_count else ""

    return _ANCHOR.sub(replace, text)


def remap_anchors(text: str, mapping: dict[int, int]) -> str:
    """Rewrite batch-local citation numbers into ledger positions.

    The analyst reads one batch numbered from 1, while the report and the appendix
    number every source collected in the turn. Without this rewrite the same anchor
    means two different pages in the two places, and a citation that survives looks
    correct while pointing at the wrong article.
    """
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return f"[{mapping[index]}]" if index in mapping else ""

    return _ANCHOR.sub(replace, text)


def batch_window(
    sources: Sequence[Source],
    analyzed: set[str],
    limit: int,
) -> tuple[list[Source], dict[int, int]]:
    """Return the next batch to analyse and its local-to-ledger index map.

    New sources are appended to the ledger, so unread pages form a tail and the
    window is always one contiguous slice. That is what makes a single offset
    enough: a non-contiguous selection would make bare numbers wrong and force
    per-source identity tokens instead.

    ``mapping`` maps 1-based batch numbers to 1-based ledger positions; it is the
    input to :func:`remap_anchors` and to the coverage normalisers.
    """
    if limit <= 0:
        return [], {}
    keys = [url_key(source.url) for source in sources]
    start = next((index for index, key in enumerate(keys) if key not in analyzed), None)
    if start is None:
        # Everything has been read already. Re-reading the tail keeps a re-analysis
        # from running on an empty window; the caller reports it as wasted work.
        start = max(len(sources) - limit, 0)
    window = list(sources[start : start + limit])
    mapping = {index + 1: start + index + 1 for index in range(len(window))}
    return window, mapping


def join_gaps(gaps: Sequence[str] | None) -> str:
    """Join gap sentences with one separator, not two.

    Models write each gap as a full sentence, so joining them verbatim produces
    "缺失。；缺乏…。。" in every rendered section.
    """
    items = [gap.strip().rstrip("。；;.，,") for gap in gaps or []]
    return "；".join(item for item in items if item) or "无"
