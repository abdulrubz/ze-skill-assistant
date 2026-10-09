"""Hermetic settings: tests never read the developer's .env (set before any
app module is imported, since settings are cached on first use)."""

import os

os.environ["ZE_AI_ENV_FILE"] = ""
for _key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PROMPTS", "LANGFUSE_PROMPT_PREFIX",
             "AS_OF_DATE", "ZE_DEV_TOKEN"):
    os.environ.pop(_key, None)
