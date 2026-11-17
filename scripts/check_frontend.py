import csv
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".archive/media/ui-redesign"
STYLE_AUDIT = """() => {
  const elements = [...document.querySelectorAll('body *')].filter(e => e.checkVisibility() && [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()));
  const sizes = elements.map((e) => parseFloat(getComputedStyle(e).fontSize));
  const weights = [...new Set(elements.map((e) => getComputedStyle(e).fontWeight))];
  return {minimumFont: Math.min(...sizes), weights, overflow: document.querySelector('#mainContent').scrollWidth > document.querySelector('#mainContent').clientWidth, viewport: innerWidth};
}"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=ROOT / "static"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    checks, errors, audits, calls = [], [], [], []
    raw = (ROOT / "static/js/evaluation-data.js").read_text(encoding="utf-8")
    data = json.loads(raw[raw.index("{") : raw.index("};") + 1])
    scenario = {"mode": "unauthorized"}

    def mock_api(route):
        path = route.request.url.removeprefix(base)
        if path == "/info":
            route.fulfill(
                status=401 if scenario["mode"] == "unauthorized" else 200,
                json={"models": ["fake"], "default_model": "fake"},
            )
        elif path.startswith("/threads"):
            route.fulfill(json={"threads": [{"thread_id": "test-history", "title": "历史调研"}]})
        elif path == "/history":
            route.fulfill(
                json={
                    "messages": [
                        {"type": "human", "content": "历史问题"},
                        {"type": "ai", "content": "检索完成"},
                        {"type": "ai", "content": data["cases"][0]["answer"]},
                    ]
                }
            )
        elif path == "/stream":
            calls.append(route.request.post_data_json)
            if scenario["mode"] == "error":
                body = 'data: {"type":"error","content":"failure"}\n\ndata: [DONE]\n\n'
            else:
                messages = ["已完成检索", data["cases"][0]["answer"]]
                body = "".join(
                    "data: "
                    + json.dumps({"type": "message", "content": {"type": "ai", "content": m}})
                    + "\n\n"
                    for m in messages
                )
                if scenario["mode"] != "truncated":
                    body += "data: [DONE]\n\n"
            route.fulfill(content_type="text/event-stream", body=body)
        elif not route.request.url.startswith(base + "/"):
            route.abort()
        else:
            route.continue_()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000}, reduced_motion="reduce", accept_downloads=True
        )
        context.route("**/*", mock_api)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)
        expect(page.locator(".archive-row")).to_have_count(5)
        page.screenshot(path=OUT / "after-home.png")
        page.locator("#sendQuestion").click()
        expect(page.locator("#toast")).to_contain_text("填写")
        page.locator(".home-card[data-prompt='0']").click()
        expect(page.locator("#question")).not_to_have_value("")
        page.locator("#sendQuestion").click()
        expect(page.locator("#connectionDialog")).to_be_visible()
        assert not calls
        scenario["mode"] = "unauthorized"
        page.locator("#connectService").click()
        expect(page.locator("#connectionResult")).to_contain_text("未授权")
        page.locator("#closeConnection").click()
        checks.append("empty input, prompt fill, disconnected and unauthorized states")
        page.locator(".primary-nav a[href='#data']").click()
        expect(page.locator("#recordsBody tr")).to_have_count(1)
        expect(page.locator("#recordsBody")).to_contain_text("暂无真实运行记录")
        expect(page.locator("#durationChart")).to_contain_text("连接服务并完成一次调研后显示")
        page.locator("#toast").wait_for(state="hidden")
        page.screenshot(path=OUT / "after-data.png")
        with page.expect_download() as event:
            page.locator("#exportData").click()
        event.value.save_as(OUT / "filtered.csv")
        with (OUT / "filtered.csv").open(encoding="utf-8-sig", newline="") as file:
            assert len(list(csv.reader(file))) == 1
        checks.append("live data empty state and header-only CSV export")
        for index, record in enumerate(data["cases"]):
            page.goto(base + "#record/" + record["id"])
            expect(page.locator("#reportView")).to_be_visible()
            if index == 0:
                page.screenshot(path=OUT / "after-report.png")
                cite = page.locator("#reportBody [data-source]").first
                target = cite.get_attribute("data-source")
                cite.click()
                expect(page.locator("#" + target)).to_be_focused()
            if record["source_count"] == 0:
                expect(page.locator("#reportSources")).to_contain_text("未采集来源")
        with page.expect_download() as event:
            page.locator("#exportReport").click()
        assert event.value.suggested_filename.endswith(".md")
        checks.append("all five reports, citation focus, no-source report, Markdown export")
        page.goto(base + "#home")
        page.locator("#question").fill("")
        for width in (390, 768, 1440):
            page.set_viewport_size({"width": width, "height": 1000})
            for route in ("home", "data", "record/ess-cell-cn-2025", "about"):
                page.goto(base + "#" + route)
                audit = page.evaluate(STYLE_AUDIT)
                assert audit["minimumFont"] >= 12 and set(audit["weights"]) <= {"400", "500"}, audit
                assert not audit["overflow"], (route, audit)
                audits.append({"route": route, **audit})
                if width == 390:
                    page.screenshot(path=OUT / f"mobile-{route.split('/')[0]}.png")
        page.set_viewport_size({"width": 390, "height": 1000})
        page.goto(base)
        page.locator("#menuToggle").click()
        expect(page.locator("#newResearch")).to_be_focused()
        page.keyboard.press("Escape")
        expect(page.locator("#menuToggle")).to_be_focused()
        page.goto(base + "#home")
        page.set_viewport_size({"width": 1440, "height": 1000})
        checks.append(
            "three viewports, keyboard drawer, minimum 12px, only 400/500, no page overflow"
        )
        scenario["mode"] = "ok"
        page.locator("#openConnection").click()
        page.locator("#connectionDialog details summary").click()
        page.locator("#serviceToken").fill("test-only-token")
        page.locator("#connectService").click()
        expect(page.locator("#connectionLabel")).to_have_text("服务已连接")
        assert "test-only-token" not in page.evaluate("JSON.stringify(localStorage)")
        page.locator("#question").fill("离线模拟调研")
        page.locator("#sendQuestion").click()
        expect(page.locator(".chat-answer")).to_have_count(1)
        page.locator("#question").fill("继续比较")
        page.locator("#question").press("Control+Enter")
        expect(page.locator(".chat-answer")).to_have_count(2)
        assert (
            len(calls) == 2
            and calls[0]["thread_id"] == calls[1]["thread_id"]
            and calls[0]["stream_tokens"] is False
        )
        page.screenshot(path=OUT / "after-chat.png")
        for mode in ("error", "truncated"):
            scenario["mode"] = mode
            page.locator("#question").fill("验证失败分支")
            page.locator("#sendQuestion").click()
            expect(page.locator(".chat-status.error").last).to_contain_text(
                "失败" if mode == "error" else "中断"
            )
            expect(page.locator("#question")).to_have_value("验证失败分支")
        checks.append(
            "mock SSE success, multi-turn identity, service error, truncated stream, preserved draft, token stays in memory"
        )
        page.goto(base + "#chat/test-history")
        expect(page.locator(".chat-question").first).to_have_text("历史问题")
        expect(page.locator(".chat-answer")).to_have_count(1)
        checks.append("server history restoration with intermediate messages collapsed")
        security = page.evaluate("""async () => {
          const {markdown} = await import('/js/answer.js');
          const div = document.createElement('div');
          div.innerHTML = markdown('<img src=x onerror=alert(1)> [x](javascript:alert(1)) [99] **text**', [], 'safe');
          return {unsafe: div.querySelectorAll('img,script,[onerror],a[href^="javascript:"]').length, buttons: div.querySelectorAll('button').length, text: div.textContent};
        }""")
        assert security["unsafe"] == 0 and security["buttons"] == 0
        checks.append("untrusted Markdown escaped; missing citations are inert")
        stream_test = page.evaluate("""async () => {
          const {streamResearch} = await import('/js/api.js');
          const original = window.fetch;
          const bytes = new TextEncoder().encode('data: {"type":"message","content":{"type":"ai","content":"中文分块"}}\\r\\n\\r\\ndata: [DONE]\\r\\n\\r\\n');
          window.fetch = async () => new Response(new ReadableStream({start(c) { for (const byte of bytes) c.enqueue(new Uint8Array([byte])); c.close(); }}), {headers:{'content-type':'text/event-stream'}});
          try { const rows = []; for await (const row of streamResearch({}, new AbortController().signal)) rows.push(row); return rows; }
          finally { window.fetch = original; }
        }""")
        assert stream_test == ["中文分块"]
        page.evaluate("""() => {
          window.originalFetch = window.fetch;
          window.fetch = (url, options) => url === '/stream' ? Promise.resolve(new Response(new ReadableStream({start(c) { options.signal.addEventListener('abort', () => c.error(new DOMException('Aborted','AbortError'))); }}), {headers:{'content-type':'text/event-stream'}})) : window.originalFetch(url, options);
        }""")
        page.locator("#question").fill("测试停止等待")
        page.locator("#sendQuestion").click()
        expect(page.locator("#stopWaiting")).to_be_visible()
        page.locator("#stopWaiting").click()
        expect(page.locator(".chat-status.error").last).to_contain_text("后端可能仍在运行")
        expect(page.locator("#question")).to_have_value("测试停止等待")
        page.evaluate("window.fetch = window.originalFetch; delete window.originalFetch")
        checks.append(
            "single-byte UTF-8 chunks, CRLF SSE framing, stop-waiting with billing caveat"
        )
        assert not errors, errors
        browser.close()
    server.shutdown()
    result = {
        "checks": checks,
        "responsiveAudits": audits,
        "browserErrors": errors,
        "mockResearchRequests": len(calls),
        "realModelRequests": 0,
    }
    (OUT / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
