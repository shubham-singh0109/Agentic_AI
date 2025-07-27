from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, client  # ✅ Import shared Chroma client
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import uuid

# === Setup primary model + prompt chain ===
model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# === Setup embedding and Chroma store for AI_prompts collection ===
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
ai_prompts_store = Chroma(
    collection_name="AI_User_Chats",
    embedding_function=embeddings,
    client=client
)

# === Interactive Loop ===
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.strip().lower() == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)

    # === Save question + answer to AI_prompts Chroma ===
    doc = Document(
        page_content=f"Q: {question}\nA: {result}",
        metadata={"source": "chat"},
    )
    doc_id = str(uuid.uuid4())  # Ensure unique IDs
    ai_prompts_store.add_documents(documents=[doc], ids=[doc_id])