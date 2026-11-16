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
        "js/data.js",
        "js/api.js",
        "js/answer.js",
        "js/connection.js",
    ):
        response = test_client.get(f"/ui/{path}")
        assert response.status_code == 200


def test_frontend_cannot_read_project_files(test_client: TestClient):
    response = test_client.get("/ui/%2e%2e/pyproject.toml")
    assert response.status_code == 404
