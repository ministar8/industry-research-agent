from fastapi.testclient import TestClient


def test_frontend_is_available_without_model_calls(test_client: TestClient, mock_agent):
    response = test_client.get("/ui/")
    assert response.status_code == 200
    assert 'lang="zh-CN"' in response.text
    assert 'id="question"' in response.text
    assert 'id="dataView"' in response.text
    mock_agent.ainvoke.assert_not_called()
    mock_agent.astream.assert_not_called()


def test_frontend_assets_are_served(test_client: TestClient):
    for path in (
        "css/base.css",
        "css/components.css",
        "css/views.css",
        "js/app.js",
        "js/evaluation-data.js",
        "js/runtime-records.js",
        "js/api.js",
        "js/answer.js",
        "js/connection.js",
    ):
        response = test_client.get(f"/ui/{path}")
        assert response.status_code == 200


def test_frontend_cannot_read_project_files(test_client: TestClient):
    response = test_client.get("/ui/%2e%2e/pyproject.toml")
    assert response.status_code == 404


def test_static_assets_force_revalidation(test_client: TestClient):
    for path in ("/ui/js/app.js", "/js/chat.js", "/"):
        response = test_client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"


def test_static_assets_revalidate_with_304(test_client: TestClient):
    first = test_client.get("/ui/js/app.js")
    assert first.status_code == 200

    second = test_client.get(
        "/ui/js/app.js", headers={"If-Modified-Since": first.headers["last-modified"]}
    )
    assert second.status_code == 304
