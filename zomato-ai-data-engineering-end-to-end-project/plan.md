# Multi-Provider LLM and Embeddings Plan

## Goal

Add a provider-neutral AI layer to the Zomato AI data engineering project so one configuration change selects the chat/response provider used by review enrichment, RAG, and text-to-SQL. Support Gemini, Groq, OpenAI GPT, and Anthropic Claude through LangChain integrations, and use LangGraph for the AI workflows where it fits.

Keep the implementation limited to the AI layer and its runtime configuration. Do not change the core data pipeline, Snowflake/dbt models, business logic, or Airflow DAG dependencies beyond what is strictly needed to pass AI settings into the existing enrichment task.

## Current state

- `ai/enrich_reviews.py` directly calls the OpenAI chat-completions API for JSON review classification.
- `ai/rag_chat.py` directly calls OpenAI for both review/question embeddings and answer generation.
- `ai/text_to_sql.py` directly calls OpenAI to generate SQL, then uses the existing SQL safety check before execution.
- `ai/example.env` and `airflow/example.env` currently document `OPENAI_API_KEY`.
- `airflow/Dockerfile` installs the OpenAI SDK; the Airflow DAG launches the enrichment script.
- There is no AI-specific requirements file in the project at present.

## Provider and embedding constraints

### Chat / response models

All four providers will be selectable through one shared setting, for example:

```env
LLM_PROVIDER=gemini
```

Supported values: `gemini`, `groq`, `openai`, and `anthropic`.

The selected provider will apply consistently to review enrichment, RAG answers, and SQL generation. Provider-specific model names will be configurable independently so model changes do not require code edits.

### Embeddings

Embedding selection must be a separate setting:

```env
EMBEDDING_PROVIDER=gemini
```

Use the supported OpenAI or Gemini embedding integration. Groq and Anthropic will remain valid response providers, but the RAG workflow will use the configured OpenAI or Gemini embedding provider; it must not pretend Groq or Claude can create embeddings through the same API.

The vector cache needs provider/model metadata or equivalent validation. If its embedding provider/model differs from the current configuration, the app must rebuild or clearly reject the stale cache rather than compare incompatible vectors.

### Free-tier expectations

Gemini and Groq may offer limited free usage subject to account eligibility, quotas, and changing provider policies. GPT/OpenAI and Claude/Anthropic should be treated as supported providers, not promised as free services; their API access may require billing. The application must report missing credentials or provider quota/API errors explicitly and must not silently switch providers.

## Proposed design

1. **Shared provider factory**
   - Add an AI-only module (proposed: `ai/providers.py`) that reads and validates provider/model environment settings.
   - Construct typed LangChain chat-model and embedding clients using the provider integrations.
   - Keep API keys in environment variables only; do not add secrets to source control.
   - Validate required credentials at startup and provide actionable errors naming the missing setting without printing key values.
   - Use explicit provider-specific configuration rather than generic OpenAI-compatible endpoints unless a provider integration requires it.

2. **Provider-neutral AI workflows**
   - Migrate the three existing OpenAI call sites to the shared LangChain interfaces.
   - Use LangGraph to compose the existing AI steps without changing their business responsibilities:
     - Enrichment: classify one review and validate the structured result before storing it.
     - RAG: embed/retrieve reviews, then generate an answer using the retrieved context.
     - Text-to-SQL: generate SQL, then retain the current safety gate before any Snowflake execution.
   - Keep Streamlit presentation, Snowflake access, RAG ranking behavior, and table contracts in their current applications. Do not let a graph bypass the current SQL validation.
   - Preserve model attribution in enrichment rows and ensure it records the actual selected model/provider.

3. **Configuration and dependencies**
   - Add an AI dependency manifest (proposed: `ai/requirements.txt`) for LangChain core, LangGraph, and the OpenAI, Google GenAI, Groq, and Anthropic integrations.
   - Update `ai/example.env` with provider selections, provider-specific model names, and the supported API key variables.
   - Update `airflow/example.env` and `airflow/Dockerfile` so the scheduled enrichment task has the same provider implementation and settings as local runs.
   - Keep Snowflake credentials and existing Airflow scheduling configuration unchanged.

4. **Safe migration and compatibility**
   - Default the initial configuration to Gemini for chat and Gemini for embeddings, subject to available account access and model availability.
   - Retain the existing OpenAI behavior as a selectable configuration.
   - Do not add automatic fallback: an invalid provider, missing key, unsupported embedding provider, malformed structured output, or provider failure should surface a clear error.
   - Do not reuse cached embeddings across different provider/model/dimension combinations.
   - Keep existing data schemas, output fields, and AI app entry points stable.

## Expected configuration

Illustrative configuration (exact model identifiers will be selected and verified during implementation):

```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini

GEMINI_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

GEMINI_CHAT_MODEL=
GROQ_CHAT_MODEL=
OPENAI_CHAT_MODEL=
ANTHROPIC_CHAT_MODEL=
GEMINI_EMBEDDING_MODEL=
OPENAI_EMBEDDING_MODEL=
```

Changing `LLM_PROVIDER` will switch all three response-generation tasks together. `EMBEDDING_PROVIDER` is deliberately independent because not every chat provider exposes an embedding API. API key variables are required only for the provider(s) selected in configuration.

## Files expected to change after approval

- Add `ai/providers.py` (shared provider configuration/factories).
- Add the provider-neutral LangGraph workflow module(s), keeping workflow boundaries small and aligned with the existing AI tasks.
- Update `ai/enrich_reviews.py`, `ai/rag_chat.py`, and `ai/text_to_sql.py` to use the shared workflows/models.
- Add `ai/requirements.txt`.
- Update `ai/example.env`, `airflow/example.env`, and `airflow/Dockerfile`.
- Update the AI setup/run and provider configuration sections in `README.md`.

No edits to dbt models, Snowflake SQL, core Airflow task ordering, or unrelated project components are planned.

## Validation plan

1. Verify provider setting parsing, supported-provider validation, required-key checks, and model selection without making paid API calls.
2. Add focused tests with mocked LangChain model/embedding clients for all four chat-provider selections and the two embedding-provider selections.
3. Test that RAG cache metadata detects provider/model changes and does not mix incompatible embeddings.
4. Test enrichment structured output parsing and preservation of the existing Snowflake output fields/model attribution.
5. Test text-to-SQL workflow behavior to confirm the existing safety check remains mandatory before query execution.
6. Run syntax/import checks and the focused test suite; verify dependency installation in the target local/Airflow environments.
7. With credentials available, run one small smoke test per configured provider. Do not run bulk enrichment or incur unapproved API costs.

## Approval checkpoint

This document is the plan only; no application code or dependency files have been changed. After approval, implementation will follow this scope. The main design limitation to accept is that chat/response provider selection is one setting, while embedding provider selection is a separate setting restricted to supported embedding APIs.
