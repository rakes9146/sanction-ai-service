"""Tests must not depend on a developer's private config/application-local.yml (keys, openai, langsmith)."""
import os

os.environ["AI_LOCAL_CONFIG"] = "false"
