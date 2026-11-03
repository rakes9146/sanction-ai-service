"""
config.py - ALL settings of the service in one place.

WHAT: `Settings` reads the PROPERTY FILES `config/application.yml` (+ your private `config/application-local.yml`), environment
      variables and a local `.env` file. `Rules` reads the four rule YAML files in `config/`.
WHY:  one object to pass around, nothing is read from the environment anywhere else, and no model name or key is written in the code.

Where a value comes from, strongest first:
    1. environment variable           (Docker / CI)        KAFKA_BOOTSTRAP=kafka:9092
    2. .env file                      (older, still works)
    3. config/application-local.yml   (YOUR machine: keys and models, never committed)
    4. config/application.yml         (the defaults, safe to commit, no secrets)
    5. the default written next to the field below (last resort)

The yml is nested for reading; a key is joined with "_" to find its field:  `kafka: {bootstrap: x}` -> `kafka_bootstrap`.

New to Python? `BaseSettings` (from pydantic-settings) turns every attribute below into an environment variable of the same
name in UPPER CASE.  Example: `kafka_bootstrap` is read from `KAFKA_BOOTSTRAP`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# <project>/config  (this file is <project>/src/ai_service/config.py)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _flatten(prefix: str, node: Any, out: dict[str, Any]) -> None:
    """{"kafka": {"bootstrap": "x"}} -> {"kafka_bootstrap": "x"}. Empty values are skipped so the next layer can fill them."""
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten(f"{prefix}_{key}" if prefix else str(key), value, out)
    elif node is not None and node != "":
        out[prefix] = node


def load_property_files() -> dict[str, Any]:
    """Reads application.yml, then application-local.yml on top of it. AI_CONFIG_DIR moves the folder; AI_LOCAL_CONFIG=false skips the local file."""
    folder = Path(os.environ.get("AI_CONFIG_DIR") or DEFAULT_CONFIG_DIR)
    names = ["application.yml"]
    if os.environ.get("AI_LOCAL_CONFIG", "true").lower() != "false":
        names.append("application-local.yml")
    merged: dict[str, Any] = {}
    for name in names:
        path = folder / name
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                _flatten("", yaml.safe_load(f) or {}, merged)
    return merged


class _PropertyFileSource(PydanticBaseSettingsSource):
    """Plugs the yml files into pydantic-settings as one more place to read values from."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return load_property_files()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        # first = strongest. Values passed in code (tests) beat env vars, env vars beat .env, .env beats the property files.
        return (init_settings, env_settings, dotenv_settings, _PropertyFileSource(settings_cls), file_secret_settings)

    # ---- which brain ----
    llm_provider: Literal["openai", "fake"] = "fake"
    embedder: Literal["openai", "hash"] = "hash"

    # ---- OpenAI: no model is built into the code, you choose and pin them ----
    openai_api_key: str = ""
    openai_model_l1: str = ""
    openai_model_l2: str = ""
    openai_embedding_model: str = ""
    openai_base_url: str = ""         # empty = api.openai.com. Set it to use a local OpenAI-compatible server (Ollama, LM Studio, vLLM ...)
    llm_temperature: float = 0.0      # the catalogue asks for 0 so the same input gives the same answer
    llm_timeout_s: float = 30.0

    # ---- LangSmith tracing (see tracing.py). Off by default. ----
    langsmith_enabled: bool = False          # same thing as `tracing` below: either one switches tracing on
    langsmith_tracing: bool = False          # yml: langsmith.tracing
    langsmith_agent_id: str = ""             # yml: langsmith.agent_id  -> the project name, e.g. sanction-ai-agent
    langsmith_agent_environment: str = ""    # yml: langsmith.agent_environment -> appended: sanction-ai-agent-local
    langsmith_api_key: str = ""
    langsmith_project: str = "sanction-ai-service"
    langsmith_endpoint: str = "https://api.smith.langchain.com"   # EU workspace: https://eu.api.smith.langchain.com
    langsmith_hide_inputs: bool = False      # True = the trace shows no prompts / payment text
    langsmith_hide_outputs: bool = False

    # ---- switches ----
    ai_enabled: bool = True                  # False = kill switch: no LLM call, result FAILED/AI_KILL_SWITCH (orchestrator applies G-08)
    ai_l2_may_clear: bool = True             # False = AI L2 never clears: every L2/L3 band goes to the officer
    ai_l2_may_clear_floor_l2: bool = False   # False = a payment with ANY floor is never cleared by the AI
    ai_qa_percent: int = -1                  # -1 = use config/qa.yml
    ai_max_concurrency: int = 4              # how many alerts are processed at the same time
    rescreen_timeout_s: float = 2.0

    # ---- infrastructure ----
    kafka_bootstrap: str = "localhost:9092"
    mongo_uri: str = "mongodb://localhost:27017/?directConnection=true"
    mongo_db: str = "ai_db"
    qdrant_path: str = "data/qdrant"         # embedded vector store, no container needed
    qdrant_url: str = ""                     # set this to use a Qdrant server instead
    server_port: int = 8083
    config_dir: Path = DEFAULT_CONFIG_DIR

    @model_validator(mode="after")
    def _langsmith_aliases(self) -> "Settings":
        """`tracing` = `enabled`; `agent_id` (+ `-agent_environment`) = the project. The short names are accepted so either style of yml works."""
        if self.langsmith_tracing:
            self.langsmith_enabled = True
        if self.langsmith_agent_id:
            self.langsmith_project = self.langsmith_agent_id + (f"-{self.langsmith_agent_environment}" if self.langsmith_agent_environment else "")
        return self

    def check_openai_ready(self) -> None:
        """Fail early with a clear message instead of failing on the first payment."""
        missing = []
        if self.llm_provider == "openai":
            for name in ("openai_api_key", "openai_model_l1", "openai_model_l2"):
                if not getattr(self, name):
                    missing.append(name.upper())
        if self.embedder == "openai":
            for name in ("openai_api_key", "openai_embedding_model"):
                if not getattr(self, name):
                    missing.append(name.upper())
        if self.langsmith_enabled and not self.langsmith_api_key:
            missing.append("LANGSMITH_API_KEY")
        if missing:
            raise ValueError("these settings are required by the chosen options but are empty: " + ", ".join(sorted(set(missing))))


class Rules(BaseModel):
    """The four YAML files in config/, loaded once. They are plain dictionaries: open the yml files to see what is in them."""

    l1_plan: dict
    scoring: dict
    floors: dict
    qa: dict

    @staticmethod
    def load(config_dir: Path = DEFAULT_CONFIG_DIR) -> "Rules":
        def read(name: str) -> dict:
            with open(config_dir / name, encoding="utf-8") as f:
                return yaml.safe_load(f)

        return Rules(l1_plan=read("l1_plan.yml"), scoring=read("scoring.yml"), floors=read("floors.yml"), qa=read("qa.yml"))

    def qa_percent(self, settings: Settings) -> int:
        return settings.ai_qa_percent if settings.ai_qa_percent >= 0 else int(self.qa["percent"])
