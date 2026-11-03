"""The yml property files feed Settings, and the layers win in the documented order."""
from ai_service.config import Settings


def write(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


def test_application_yml_values_are_read_and_nested_keys_are_joined(tmp_path, monkeypatch):
    write(tmp_path, "application.yml", "kafka:\n  bootstrap: broker:1\nlangsmith:\n  enabled: true\n  api_key: lsv2_x\nopenai:\n  base_url: http://localhost:11434/v1\n")
    monkeypatch.setenv("AI_CONFIG_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.kafka_bootstrap == "broker:1"
    assert s.langsmith_enabled is True and s.langsmith_api_key == "lsv2_x"
    assert s.openai_base_url == "http://localhost:11434/v1"


def test_local_file_beats_the_default_file_and_env_beats_both(tmp_path, monkeypatch):
    write(tmp_path, "application.yml", "llm:\n  provider: fake\nmongo:\n  db: from_default\n")
    write(tmp_path, "application-local.yml", "llm:\n  provider: openai\nmongo:\n  db: from_local\n")
    monkeypatch.setenv("AI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("AI_LOCAL_CONFIG", "true")
    monkeypatch.setenv("MONGO_DB", "from_env")
    s = Settings(_env_file=None)
    assert s.llm_provider == "openai"          # local over default
    assert s.mongo_db == "from_env"            # environment over files


def test_blank_values_fall_through_to_the_code_default(tmp_path, monkeypatch):
    write(tmp_path, "application.yml", "openai:\n  api_key:\nserver:\n  port:\n")
    monkeypatch.setenv("AI_CONFIG_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.openai_api_key == "" and s.server_port == 8083


def test_langsmith_on_without_a_key_is_refused_at_start_up(tmp_path, monkeypatch):
    write(tmp_path, "application.yml", "langsmith:\n  enabled: true\n")
    monkeypatch.setenv("AI_CONFIG_DIR", str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        Settings(_env_file=None).check_openai_ready()


def test_reasoning_models_get_no_temperature_argument():
    from ai_service.llm.openai_llm import _temperature_arg

    assert _temperature_arg("gpt-5", 0.0) == {}
    assert _temperature_arg("o3-mini", 0.0) == {}
    assert _temperature_arg("gpt-4.1", 0.0) == {"temperature": 0.0}


def test_langsmith_tracing_agent_id_and_environment_are_accepted(tmp_path, monkeypatch):
    write(tmp_path, "application.yml", "langsmith:\n  tracing: true\n  api_key: lsv2_x\n  agent_id: sanction-ai-agent\n  agent_environment: local\n")
    monkeypatch.setenv("AI_CONFIG_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.langsmith_enabled is True
    assert s.langsmith_project == "sanction-ai-agent-local"
    s.check_openai_ready()                       # key present -> fine
