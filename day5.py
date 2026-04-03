import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. Load API Key
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("API_KEY")

# ==========================================
# 2. Setup Models + DB
# ==========================================
DB_NAME = "vector_db"
#error handling for missing db
def check_db_exists():
    if not os.path.exists(DB_NAME):
        return False, "⚠️ Vector database not found. Please run the ingestion script first."
    return True, "✅ Database loaded successfully."
db_ok, db_status = check_db_exists()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ==========================================
# 3. Manual Memory
# ==========================================
memory_buffer = []

def update_memory(user, bot):
    memory_buffer.append((user, bot))
    if len(memory_buffer) > 4:
        memory_buffer.pop(0)

def get_memory_text():
    return "\n".join(
        [f"User: {u}\nAssistant: {b}" for u, b in memory_buffer]
    )

# ==========================================
# 4. Build BM25 Index
# ==========================================
def build_bm25_index():
    data = vectorstore._collection.get(include=["documents", "metadatas"])

    docs = data["documents"]
    metas = data["metadatas"]

    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    return bm25, docs, metas
if db_ok:
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    bm25, bm25_docs, bm25_meta = build_bm25_index()
else:
    vectorstore = None
    bm25 = bm25_docs = bm25_meta = None
# ==========================================
# 5. Query Rewriting 
# ==========================================
def rewrite_query(query):
    history = get_memory_text()

    prompt = f"""
Rewrite the user's question into a clear standalone question.

Conversation:
{history}

Question:
{query}

Rewritten standalone question:
"""

    return llm.invoke(prompt).content.strip()

# ==========================================
# 6. BM25 Search
# ==========================================
def bm25_search(query, k=6, filter_type=None):
    if bm25 is None:
        return []
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    ranked = np.argsort(scores)[::-1]

    results = []
    for idx in ranked:
        if filter_type and bm25_meta[idx]["doc_type"] != filter_type:
            continue

        results.append({
            "content": bm25_docs[idx],
            "score": scores[idx]
        })

        if len(results) >= k:
            break

    return results

# ==========================================
# 7. Vector Search
# ==========================================
def vector_search(query, k=6, filter_type=None):
    if vectorstore is None:
        return []
    if filter_type:
        docs = vectorstore.similarity_search(
            query,
            k=k,
            filter={"doc_type": filter_type}
        )
    else:
        docs = vectorstore.similarity_search(query, k=k)

    return [{"content": d.page_content, "score": 1.0} for d in docs]

# ==========================================
# 8. Hybrid Search 
# ==========================================
def hybrid_search(query, k=12, filter_type=None):
    bm25_res = bm25_search(query, k, filter_type)
    vec_res = vector_search(query, k, filter_type)

    combined = {}

    # More weight to semantic search
    for r in bm25_res:
        combined[r["content"]] = combined.get(r["content"], 0) + r["score"] * 0.4

    for r in vec_res:
        combined[r["content"]] = combined.get(r["content"], 0) + 0.6

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return [text for text, _ in ranked[:k]]

# ==========================================
# 9. Chat Function 
# ==========================================
def chat(message, history, doc_filter=None):
    if not db_ok:
        return "❌ The knowledge base is not available. Please set up the database first."

    if not message.strip():
        return "Please enter a question."
    # Step 1: Rewrite query (fixes follow-ups)
    standalone_query = rewrite_query(message)

    # Step 2: Retrieve better context
    retrieved_docs = hybrid_search(
        standalone_query,
        k=12,
        filter_type=doc_filter
    )
    if not retrieved_docs or all(len(doc.strip()) < 20 for doc in retrieved_docs):
        return "I couldn't find relevant information in the InsureLLM knowledge base for your question."

    context = "\n\n".join(retrieved_docs)

    # Step 3: Inject memory
    history_text = get_memory_text()

    system_prompt = f"""
You are an expert assistant for Insurellm.

Conversation so far:
{history_text}

Use the context below to answer accurately.
If information is missing, say so clearly.

Context:
{context}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message)
    ]

    # Step 4: Generate response
    response = llm.invoke(messages).content

    # Step 5: Update memory
    update_memory(message, response)

    return response