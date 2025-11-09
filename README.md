This is a Retrieval-Augmented Generation (RAG) app built with:
- Cohere (for embeddings + chat)
- Pinecone (for vector DB)
- Streamlit (for UI)


- Upload and query text, PDFs, or tabular data.
- Semantic search and natural-language answers.
- Optional NL→SQL querying.

### Run locally ###
```bash
pip install -r requirements.txt
streamlit run app.py
