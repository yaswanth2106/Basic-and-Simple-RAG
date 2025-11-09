import streamlit as st
from dotenv import load_dotenv
import os
import cohere
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import sqlite3
from PyPDF2 import PdfReader
import io
import uuid

# ──────────────── SETUP ────────────────
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not COHERE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ Missing COHERE_API_KEY or PINECONE_API_KEY in .env file.")
    st.stop()

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

EMBED_MODEL = "embed-english-v3.0"
CHAT_MODEL = "command-r-plus-08-2024"
SIM_THRESHOLD = 0.00000125


# ──────────────── HELPERS ────────────────
def embed_and_store(texts, metadata_tag="general"):
    """Embed and store texts into Pinecone."""
    embeds = co.embed(texts=texts, model=EMBED_MODEL, input_type="search_document").embeddings
    vectors = [
        (f"{metadata_tag}-{uuid.uuid4().hex}", embeds[i], {"tag": metadata_tag, "text": texts[i]})
        for i in range(len(texts))
    ]
    index.upsert(vectors=vectors)


def read_pdf(file):
    """Extract text from PDF."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def read_table(file):
    """Try reading CSV, TSV, or Excel file into DataFrame."""
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith(".tsv"):
            return pd.read_csv(file, sep="\t")
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
        else:
            st.error("Unsupported table format.")
            return None
    except Exception as e:
        st.error(f"Failed to read table: {e}")
        return None


def df_to_text_rows(df: pd.DataFrame, table_name="uploaded_table"):
    """Convert table rows to plain-text chunks."""
    rows = []
    for i, row in df.iterrows():
        txt = "; ".join([f"{c}: {row[c]}" for c in df.columns])
        rows.append(f"Table {table_name}, Row {i}: {txt}")
    return rows


def query_rag(query, tag_filter=None, top_k=5):
    """Retrieve and answer using Cohere chat."""
    q_embed = co.embed(texts=[query], model=EMBED_MODEL, input_type="search_query").embeddings[0]
    filt = {"tag": {"$eq": tag_filter}} if tag_filter else None
    results = index.query(vector=q_embed, top_k=top_k, include_metadata=True, filter=filt)

    matches = [m for m in results["matches"] if m["score"] >= SIM_THRESHOLD]
    if not matches:
        return "⚠️ No relevant info found.", []

    contexts = [m["metadata"]["text"] for m in matches]
    context_text = "\n".join(contexts)

    prompt = f"""Answer the question only using the context below.
If the context is unrelated, say you don't know.

Context:
{context_text}

Question: {query}

Answer:"""

    resp = co.chat(model=CHAT_MODEL, message=prompt, preamble="You are a helpful data assistant.", temperature=0.6)
    return resp.text, matches


# ──────────────── STREAMLIT UI ────────────────
st.title("🧠 Multi-Modal RAG — Upload Text / PDF / Table & Query")

uploaded_file = st.file_uploader("Upload a text (.txt), PDF, or table (.csv/.xlsx/.tsv) file", type=["txt", "pdf", "csv", "tsv", "xlsx"])
text_input = st.text_area("Or paste raw text manually here:")

mode = st.radio("Choose mode:", ["Ingest File/Text", "Ask a Question"])

# ──────────────── INGESTION ────────────────
if mode == "Ingest File/Text":
    if uploaded_file is not None:
        filename = uploaded_file.name
        if filename.endswith(".pdf"):
            text = read_pdf(uploaded_file)
            if text:
                st.success("✅ Extracted text from PDF.")
                st.text_area("Preview", text[:1000])
                if st.button("Embed & Store PDF"):
                    embed_and_store([text], metadata_tag=filename)
                    st.success("✅ Stored PDF content in Pinecone.")
        elif filename.endswith((".csv", ".tsv", ".xlsx")):
            df = read_table(uploaded_file)
            if df is not None:
                st.dataframe(df.head())
                if st.button("Embed & Store Table Rows"):
                    texts = df_to_text_rows(df, filename)
                    embed_and_store(texts, metadata_tag=filename)
                    st.success("✅ Stored table rows in Pinecone.")
        elif filename.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            if st.button("Embed & Store Text File"):
                embed_and_store([text], metadata_tag=filename)
                st.success("✅ Stored text file in Pinecone.")
        else:
            st.error("Unsupported file type.")

    elif text_input.strip():
        if st.button("Embed & Store Typed Text"):
            embed_and_store([text_input.strip()], metadata_tag="manual-text")
            st.success("✅ Stored typed text in Pinecone.")
    else:
        st.info("Upload a file or paste text to start ingestion.")

# ──────────────── QUERY ────────────────
elif mode == "Ask a Question":
    query = st.text_input("Ask a question about your uploaded data:")
    tag_filter = st.text_input("(Optional) Filter by filename/tag:")
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                answer, matches = query_rag(query.strip(), tag_filter or None)
            st.markdown("### 🧠 Answer:")
            st.write(answer)
            if matches:
                st.markdown("### 🔍 Top retrieved chunks:")
                for m in matches:
                    st.caption(f"({m['score']:.3f}) {m['metadata']['text'][:200]}...")
