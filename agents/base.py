from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def get_anthropic_client():
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")
    return anthropic.Anthropic(api_key=api_key)
