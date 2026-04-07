import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. Setup and Configurations
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("API_KEY")
DB_NAME = "vector_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

# Initialize Database and BM25
if os.path.exists(DB_NAME):
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    data = vectorstore._collection.get(include=["documents", "metadatas"])
    bm25_docs, bm25_meta = data["documents"], data["metadatas"]
    bm25 = BM25Okapi([d.lower().split() for d in bm25_docs])
else:
    vectorstore = bm25 = None

# ==========================================
# 2. Search Logic (Hybrid RAG)
# ==========================================
def rewrite_query(query, history):
    if not history: return query
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = f"Rewrite this into a standalone question:\n\nChat:\n{history_text}\n\nQuestion: {query}"
    return llm.invoke(prompt).content.strip()

def hybrid_search(query, k=6, filter_type=None):
    if not vectorstore: return []
    
    # Vector Search
    v_search = vectorstore.similarity_search(query, k=k, filter={"doc_type": filter_type} if filter_type else None)
    vec_res = [{"content": d.page_content, "rank": i} for i, d in enumerate(v_search)]
    
    # BM25 Search
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1]
    bm25_res = []
    for idx in ranked:
        if filter_type and bm25_meta[idx]["doc_type"] != filter_type: continue
        bm25_res.append({"content": bm25_docs[idx], "rank": len(bm25_res)})
        if len(bm25_res) >= k: break

    # Re-ranking (RRF)
    scores = {}
    for r in bm25_res + vec_res:
        scores[r["content"]] = scores.get(r["content"], 0) + 1 / (60 + r["rank"])
    
    ranked_final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked_final[:k]]

# ==========================================
# 3. Generation Logic
# ==========================================
def chat(message, history, doc_filter=None):
    if not vectorstore: return "⚠️ Database not found."
    
    query = rewrite_query(message, history)
    docs = hybrid_search(query, k=6, filter_type=doc_filter)
    
    if not docs:
        return "I'm a chatbot for InsureLLM. I only answer questions about our employees and products."

    context = "\n\n".join(docs)
    system_prompt = f"You are an InsureLLM assistant. Use this context:\n{context}\n\nOnly answer about InsureLLM."
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    response = llm.invoke(messages)
    return response.content