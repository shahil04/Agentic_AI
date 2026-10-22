import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from providers import (  # noqa: E402
    AISettings,
    create_chat_model,
    create_embedding_model,
    embedding_cache_matches,
    load_settings,
)
from workflows import invoke_json_prompt, invoke_prompt  # noqa: E402


class ProviderConfigurationTests(unittest.TestCase):
    def test_defaults_select_gemini_for_chat_and_embeddings(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()

        self.assertEqual(settings.llm_provider, "gemini")
        self.assertEqual(settings.embedding_provider, "gemini")
        self.assertEqual(settings.model_identifier, "gemini/gemini-2.5-flash")

    def test_selects_each_supported_chat_provider(self):
        for provider, model in (
            ("gemini", "gemini-2.5-flash"),
            ("groq", "llama-3.3-70b-versatile"),
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-5-haiku-latest"),
        ):
            with self.subTest(provider=provider), patch.dict(
                os.environ, {"LLM_PROVIDER": provider}, clear=True
            ):
                settings = load_settings()
                self.assertEqual(settings.llm_provider, provider)
                self.assertEqual(settings.llm_model, model)

    def test_rejects_unknown_chat_and_embedding_providers(self):
        for variable, value in (
            ("LLM_PROVIDER", "unknown"),
            ("EMBEDDING_PROVIDER", "groq"),
        ):
            with self.subTest(variable=variable), patch.dict(
                os.environ, {variable: value}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, variable):
                    load_settings()

    def test_reports_missing_selected_provider_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}, clear=True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                create_chat_model()

    def test_embedding_cache_requires_matching_provider_and_model_metadata(self):
        columns = {"_embedding_provider", "_embedding_model", "embedding"}
        self.assertTrue(
            embedding_cache_matches(
                columns,
                ["gemini", "gemini"],
                ["embedding-v1", "embedding-v1"],
                "gemini",
                "embedding-v1",
            )
        )
        self.assertFalse(
            embedding_cache_matches(
                columns,
                ["openai"],
                ["embedding-v1"],
                "gemini",
                "embedding-v1",
            )
        )
        self.assertFalse(
            embedding_cache_matches(
                {"embedding"},
                [],
                [],
                "gemini",
                "embedding-v1",
            )
        )

    def test_chat_factory_uses_selected_integration_and_model(self):
        class FakeChatModel:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        for provider, module_name, class_name, key_name in (
            ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI", "GEMINI_API_KEY"),
            ("groq", "langchain_groq", "ChatGroq", "GROQ_API_KEY"),
            ("openai", "langchain_openai", "ChatOpenAI", "OPENAI_API_KEY"),
            ("anthropic", "langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"),
        ):
            with self.subTest(provider=provider):
                fake_module = types.ModuleType(module_name)
                setattr(fake_module, class_name, FakeChatModel)
                settings = AISettings(provider, "test-model", "gemini", "test-embedding")
                with patch.dict(
                    os.environ, {key_name: "test-key"}, clear=True
                ), patch.dict(sys.modules, {module_name: fake_module}):
                    model = create_chat_model(temperature=0.25, settings=settings)
                self.assertIsInstance(model, FakeChatModel)
                self.assertEqual(model.model, "test-model")
                self.assertEqual(model.temperature, 0.25)

    def test_embedding_factory_uses_gemini_or_openai(self):
        class FakeEmbeddingModel:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        for provider, module_name, class_name, key_name in (
            ("gemini", "langchain_google_genai", "GoogleGenerativeAIEmbeddings", "GEMINI_API_KEY"),
            ("openai", "langchain_openai", "OpenAIEmbeddings", "OPENAI_API_KEY"),
        ):
            with self.subTest(provider=provider):
                fake_module = types.ModuleType(module_name)
                setattr(fake_module, class_name, FakeEmbeddingModel)
                settings = AISettings("gemini", "test-chat", provider, "test-embedding")
                with patch.dict(
                    os.environ, {key_name: "test-key"}, clear=True
                ), patch.dict(sys.modules, {module_name: fake_module}):
                    model = create_embedding_model(settings=settings)
                self.assertIsInstance(model, FakeEmbeddingModel)
                self.assertEqual(model.model, "test-embedding")


class LangGraphWorkflowTests(unittest.TestCase):
    def test_runs_plain_response_through_graph(self):
        fake_model = SimpleNamespace(
            invoke=lambda messages: SimpleNamespace(content="provider answer")
        )
        with patch("workflows.create_chat_model", return_value=fake_model):
            answer = invoke_prompt("system", "question", temperature=0.2)

        self.assertEqual(answer, "provider answer")

    def test_parses_json_object_inside_graph(self):
        fake_model = SimpleNamespace(
            invoke=lambda messages: SimpleNamespace(content='{"sql":"SELECT 1"}')
        )
        with patch("workflows.create_chat_model", return_value=fake_model):
            result = invoke_json_prompt("system", "question")

        self.assertEqual(result, {"sql": "SELECT 1"})

    def test_rejects_malformed_json_from_provider(self):
        fake_model = SimpleNamespace(
            invoke=lambda messages: SimpleNamespace(content="not json")
        )
        with patch("workflows.create_chat_model", return_value=fake_model):
            with self.assertRaises(ValueError):
                invoke_json_prompt("system", "question")


if __name__ == "__main__":
    unittest.main()
