import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from dotenv import load_dotenv
from providers import create_embedding_model, embedding_cache_matches, load_settings
from workflows import invoke_prompt

load_dotenv(Path(__file__).with_name(".env"))

SETTINGS = load_settings()
EMBEDDING_PROVIDER = SETTINGS.embedding_provider
EMBEDDING_MODEL = SETTINGS.embedding_model
CHAT_MODEL = SETTINGS.model_identifier
NEW_REVIEWS = 500
TOK_K = 5
CACHE_FILE = "review_embeddings.parquet"
def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df

@st.cache_resource
def get_embedding_client(provider, model):
    settings = load_settings()
    if (settings.embedding_provider, settings.embedding_model) != (provider, model):
        raise ValueError("Embedding configuration changed; restart the Streamlit app.")
    return create_embedding_model(settings)


def embed_documents(texts, provider, model):
    if not texts:
        return []
    return get_embedding_client(provider, model).embed_documents(texts)


def embed_question(text, provider, model):
    return get_embedding_client(provider, model).embed_query(text)
    
@st.cache_data()
def load_reviews(provider, model):
    if os.path.exists(CACHE_FILE):
        cached = pd.read_parquet(CACHE_FILE)
        if embedding_cache_matches(
            set(cached.columns),
            cached.get("_embedding_provider", []),
            cached.get("_embedding_model", []),
            provider,
            model,
        ):
            return cached
        st.warning(
            "The saved review embeddings use a different or unknown model; rebuilding them."
        )

    df = read_reviews_from_snowflake()
    df["embedding"] = embed_documents(df["comment"].tolist(), provider, model)
    df["_embedding_provider"] = provider
    df["_embedding_model"] = model
    df.to_parquet(CACHE_FILE)
    return df

st.title("Chat with your Zomato Reviews")
st.caption(
    f"Searching {NEW_REVIEWS} reviews with {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL}; "
    f"answering with {CHAT_MODEL}"
)

def consine_simiarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def find_similar_reviews(question, df):
    question_vector = embed_question(question, EMBEDDING_PROVIDER, EMBEDDING_MODEL)

    scores = []
    for review_vector in df['embedding']:
        scores.append(consine_simiarity(question_vector, review_vector))

    df = df.copy()
    df['score'] = scores
    return df.nlargest(TOK_K, 'score')

def ask_llm(question, top_reviews):
    conext = ""

    for _, row in top_reviews.iterrows():
        conext += f" ({row['city']}, {row['rating']} stars) {row['comment']}\n"

    system_prompt = (
        "Answer ONLY using the customer reviews provided. "
        "Be concise. If the reviews don't covert it, say No"
    )

    user_prompt = f"Questions: {question}\n\nReviews:\n{conext}"

    return invoke_prompt(system_prompt, user_prompt, temperature=0.2)
    
review_df = load_reviews(EMBEDDING_PROVIDER, EMBEDDING_MODEL)

question = st.text_input("Ask a question about your reviews:",
                         placeholder="e.g. What are the most common complaints about delivery?")

if question:
    top_reviews = find_similar_reviews(question, review_df)
    answer = ask_llm(question, top_reviews)

    st.markdown(f"**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(top_reviews[['city', 'rating', 'comment']], hide_index=True)