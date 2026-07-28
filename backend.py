import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
DB_NAME = "vector_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

if os.path.exists(DB_NAME):
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    data = vectorstore._collection.get(include=["documents", "metadatas"])
    bm25_docs, bm25_meta = data["documents"], data["metadatas"]
else:
    vectorstore = bm25_docs = bm25_meta = None

def rewrite_query(query, history):
    if not history: return query
    recent_history = history[-3:]  # Fix 3: Bound history length
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])
    prompt = f"Rewrite this into a standalone search query:\n\nChat:\n{history_text}\n\nQuestion: {query}"
    return llm.invoke(prompt).content.strip()

def hybrid_search(query, k=6, filter_type=None):
    if not vectorstore: return []
    
    # Vector Search
    v_search = vectorstore.similarity_search(query, k=k, filter={"doc_type": filter_type} if filter_type else None)
    vec_res = [{"content": d.page_content, "rank": i} for i, d in enumerate(v_search)]
    
    # Filter BM25 Candidates
    if filter_type:
        filtered_indices = [i for i, meta in enumerate(bm25_meta) if meta.get("doc_type") == filter_type]
        candidate_docs = [bm25_docs[i] for i in filtered_indices]
    else:
        candidate_docs = bm25_docs

    if candidate_docs:
        temp_bm25 = BM25Okapi([d.lower().split() for d in candidate_docs])
        scores = temp_bm25.get_scores(query.lower().split())
        ranked_indices = np.argsort(scores)[::-1][:k]
        bm25_res = [{"content": candidate_docs[idx], "rank": i} for i, idx in enumerate(ranked_indices)]
    else:
        bm25_res = []

    # Reciprocal Rank Fusion
    scores_dict = {}
    for r in bm25_res + vec_res:
        scores_dict[r["content"]] = scores_dict.get(r["content"], 0) + 1 / (60 + r["rank"])
    
    ranked_final = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked_final[:k]]

def chat(message, history, doc_filter=None):
    if not vectorstore: return "⚠️ Database not found."
    
    query = rewrite_query(message, history)
    docs = hybrid_search(query, k=6, filter_type=doc_filter)
    
    if not docs:
        return "I cannot find any relevant information in the InsureLLM database."

    context = "\n\n".join(docs)
    
    # Fix 2: Grounding prompt against hallucinations
    system_prompt = (
        "You are an InsureLLM internal assistant.\n"
        "Answer the question strictly using ONLY the context provided below.\n"
        "If the answer cannot be found in the context, state 'I cannot find this information in our knowledge base.'\n\n"
        f"Context:\n{context}"
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    response = llm.invoke(messages)
    return response.content