from unittest.mock import patch

from agents.safeguard import Safeguard, SafetyAssessment
from schema.models import DeepseekModelName


def test_safeguard_skipped_under_fake_model():
    with (
        patch("agents.safeguard.settings.USE_FAKE_MODEL", True),
        patch("agents.safeguard.settings.DEFAULT_MODEL", DeepseekModelName.DEEPSEEK_FLASH),
        patch("agents.safeguard.get_model") as get_model,
    ):
        safeguard = Safeguard()

    get_model.assert_not_called()
    assert safeguard.model is None
    assert safeguard.invoke([]).safety_assessment == SafetyAssessment.SAFE
