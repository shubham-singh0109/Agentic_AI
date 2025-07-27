import os
import pandas as pd
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# === Step 1: Load your CSV ===
df = pd.read_csv("realistic_restaurant_reviews.csv")

# === Step 2: Initialize Ollama embeddings ===
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# === Step 3: Connect to Chroma Cloud using HttpClient ===
client = chromadb.HttpClient(
    ssl=True,
    host='api.trychroma.com',
    tenant='05aa9bac-728c-4821-a88d-34153e48bcdf',
    database='vector_store',
    headers={
        'x-chroma-token': 'ck-3cgGCDi9TmXEjEQCsBivV2wXwSsN3qRQDuEAiHQtycxf'
    }
)

# === Step 4: Collection configuration ===
collection_name = "restaurant_reviews"
existing_collections = [col.name for col in client.list_collections()]
add_documents = True

# === Step 5: Prepare documents and IDs ===
if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]}
        )
        documents.append(document)
        ids.append(str(i))

# === Step 6: Initialize the Chroma vector store ===
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    client=client
)

# === Step 7: Add documents in batches of 1000 (Chroma Cloud limit) ===
if add_documents:
    batch_size = 300
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)

# === Step 8: Create retriever ===
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Export Chroma client so it can be reused
__all__ = ["retriever", "client"]