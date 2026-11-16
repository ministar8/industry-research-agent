"""Voice input/output module.

Provides speech-to-text and text-to-speech capabilities with support
for multiple providers. These handlers are standalone and are not wired
into any UI.

Modules:
    - SpeechToText: STT handler
    - TextToSpeech: TTS handler

Quick Start:
    >>> from voice import SpeechToText, TextToSpeech
    >>> stt = SpeechToText(provider="openai")
    >>> tts = TextToSpeech(provider="openai", voice="nova")
"""

from voice.stt import SpeechToText
from voice.tts import TextToSpeech

__all__ = ["SpeechToText", "TextToSpeech"]
