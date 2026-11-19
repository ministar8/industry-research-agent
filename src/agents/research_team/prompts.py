"""Prompt loading for the research graph nodes.

Prompt bodies live in `prompts/`; this module is the only path that reads them.
"""

from pathlib import Path
from typing import cast

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate

# Prompt bodies live next to the repo root, three levels above this module, so the
# same path resolves from src/ locally and from /app inside the container.
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def read_prompt(name: str) -> str:
    """Load a prompt body from prompts/<dotted name as path>.md.

    Reading happens at import time: a missing or renamed file stops the process
    instead of failing one request later.
    """
    path = PROMPTS_DIR / (name.replace(".", "/") + ".md")
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template file is missing: {path}")
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        raise ValueError(f"Prompt template file is empty: {path}")
    return body


def render_prompt(template: ChatPromptTemplate, **values: str) -> list[AnyMessage]:
    """Render a prompt only when its declared variables match the call site."""
    expected = set(template.input_variables)
    provided = set(values)
    missing = expected - provided
    unexpected = provided - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing variables: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected variables: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Prompt variables do not match ({'; '.join(details)})")
    return cast(list[AnyMessage], template.format_messages(**values))


def load_prompt(name: str, *parts: str) -> ChatPromptTemplate:
    """Load a prompt from files named after it, one system message per file.

    Without `parts` the file is `<name>.md`; a multi-message prompt names each file
    explicitly so no prompt text stays in Python.
    """
    files = parts or (name,)
    return ChatPromptTemplate.from_messages([("system", read_prompt(file)) for file in files])
