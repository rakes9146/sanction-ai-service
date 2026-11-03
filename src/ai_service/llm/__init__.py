"""
llm/ - everything about talking to the language model.

    base.py        the LLM interface (what the graph needs) + the two request shapes
    prompts.py     the system prompts and the fenced "untrusted data" block (prompt-injection defence)
    openai_llm.py  the real model (structured output, retries)
    fake_llm.py    FakeLLM (rule-based demo, NOT AI) and ScriptedLLM (tests)
"""
from ai_service.config import Settings
from ai_service.llm.base import LLM


def make_llm(settings: Settings) -> LLM:
    """Pick the implementation from settings.LLM_PROVIDER."""
    if settings.llm_provider == "openai":
        from ai_service.llm.openai_llm import OpenAILLM

        return OpenAILLM(settings)
    from ai_service.llm.fake_llm import FakeLLM

    return FakeLLM()
