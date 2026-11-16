"""真实后端端到端联调：浏览器直连 :8080，USE_FAKE_MODEL=true，零模型/检索费用。"""

import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/ui/"


def main():
    results, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000}, accept_downloads=True
        )
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(BASE + "#home")
        page.wait_for_selector("#composer")
        results.append("首页加载")

        # 未连接时点发送 -> 弹出连接对话框
        page.fill("#question", "储能电芯 2025 出货排名与价格走势")
        page.click("#sendQuestion")
        page.wait_for_selector("#connectionDialog[open]")
        results.append("未连接时发送弹出连接框")

        # 错误口令：仅在服务配置了鉴权时才应 401；本机未配置 AUTH_SECRET 时任何口令都会成功
        page.fill("#serviceToken", "wrong-token")
        page.click("#connectService")
        page.wait_for_function(
            "document.getElementById('connectionResult').textContent.includes('已连接') || document.getElementById('connectionResult').classList.contains('error')",
            timeout=15000,
        )
        if "error" in (page.get_attribute("#connectionResult", "class") or ""):
            assert "未授权" in page.inner_text("#connectionResult"), page.inner_text(
                "#connectionResult"
            )
            results.append("错误口令返回未授权（服务已配置鉴权）")
            page.fill("#serviceToken", "")
            page.click("#connectService")
        else:
            results.append("服务未配置鉴权，错误口令也被接受（已跳过 401 断言）")
            if page.evaluate("document.getElementById('connectionDialog').open"):
                page.click("#closeConnection")
        page.wait_for_function(
            "document.getElementById('connectionLabel').textContent === '服务已连接'"
        )
        assert "wrong-token" not in page.evaluate("JSON.stringify(localStorage)")
        results.append("连接成功且口令未写入本地存储")

        # 发起调研（假模型）
        page.fill("#question", "储能电芯 2025 年出货排名与价格走势")
        page.click("#sendQuestion")
        page.wait_for_selector(".chat-answer", timeout=60000)
        assert page.locator(".chat-question").count() >= 1
        assert "没能从这段输入里识别出可调研的对象" in page.inner_text(".chat-answer")
        results.append("发送并收到假模型回答")

        # 会话列表出现新会话
        page.wait_for_selector("#historyNav a", timeout=15000)
        assert page.locator("#historyNav a").count() >= 1
        results.append("会话列表刷新")

        # 同一会话继续追问
        page.fill("#question", "继续补充说明时间范围")
        page.click("#sendQuestion")
        page.wait_for_function(
            "document.querySelectorAll('.chat-answer').length >= 2", timeout=60000
        )
        results.append("同会话多轮追问")

        thread_href = page.locator("#historyNav a").first.get_attribute("href")
        assert thread_href and thread_href.startswith("#chat/"), thread_href

        # 断开：口令只在内存，刷新后应失联
        page.click("#openConnection")
        page.click("#disconnectService")
        page.wait_for_function(
            "document.getElementById('connectionLabel').textContent === '连接调研服务'"
        )
        page.reload()
        page.wait_for_selector("#composer")
        assert page.inner_text("#connectionLabel") == "连接调研服务"
        results.append("断开后刷新页面仍为未连接（口令不落盘）")

        # 重连并读回历史会话
        page.click("#openConnection")
        page.click("#connectService")
        page.wait_for_function(
            "document.getElementById('connectionLabel').textContent === '服务已连接'"
        )
        page.goto(BASE + thread_href)
        page.wait_for_selector(".chat-answer", timeout=30000)
        assert page.locator(".chat-question").count() >= 2, page.locator(".chat-question").count()
        results.append("断开重连后历史会话可读回")

        # 假模型不检索 -> 来源栏为空态
        assert (
            page.locator("#chatSources .empty-state").count() == 1
            or page.locator("#chatSources .src-group").count() == 0
        )
        results.append("无来源时来源栏显示空态")

        page.screenshot(path="C:/Users/star/AppData/Local/Temp/ui-peek/live-e2e-chat.png")
        assert not errors, errors
        browser.close()

    print("PASS:")
    for r in results:
        print("  -", r)


main()
