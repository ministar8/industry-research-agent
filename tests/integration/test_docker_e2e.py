import os

import pytest

from client import AgentClient

# Same override as tests/smoke: point at the container under test instead of
# hardcoding a port that the CI workflow has to keep in sync.
SERVICE_URL = os.environ.get("SMOKE_SERVICE_URL", "http://localhost:8080")


@pytest.mark.docker
def test_service_with_fake_model():
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    client = AgentClient(SERVICE_URL, agent="research-supervisor")
    response = client.invoke("Tell me a joke?", model="fake")
    assert response.type == "ai"
    # research-supervisor parses intent; off-topic input hits the no-target hint, not an echo.
    assert response.content.startswith("没能从这段输入里识别出可调研的对象")
