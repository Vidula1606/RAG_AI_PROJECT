import os
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. Configuration - Path Logic
# ==========================================
BASE_DIR = Path(__file__).parent.resolve()
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge-base"
DB_NAME = str(BASE_DIR / "advanced_vector_db")

class Chunk(BaseModel):
    headline: str = Field(description="A brief heading")
    summary: str = Field(description="A brief summary")
    original_text: str = Field(description="Original text unchanged")

class Chunks(BaseModel):
    chunks: list[Chunk]

# Initialize local Llama 3 and Embeddings
llm = ChatOllama(model="llama3", temperature=0)
llm_structured = llm.with_structured_output(Chunks)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 2. RAG Chat Function (For final_project.py)
# ==========================================
def chat(message, history):
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    docs = vectorstore.similarity_search(message, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"Use this context to answer: {context}"
    messages = [SystemMessage(content=system_prompt)]
    
    for msg in history:
        content = msg.get("content") if isinstance(msg, dict) else msg
        messages.append(HumanMessage(content=content) if history.index(msg) % 2 == 0 else SystemMessage(content=content))

    messages.append(HumanMessage(content=message))
    return llm.invoke(messages).content

# ==========================================
# 3. The "Action" Functions (Building the DB)
# ==========================================
def fetch_documents():
    """Finds and loads your markdown files."""
    documents = []
    print(f"🔍 Searching in: {KNOWLEDGE_BASE_PATH}")
    
    if not KNOWLEDGE_BASE_PATH.exists():
        print("❌ FOLDER NOT FOUND!")
        return []

    for file_path in KNOWLEDGE_BASE_PATH.rglob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append({
                        "type": file_path.parent.name, 
                        "source": file_path.name, 
                        "text": text
                    })
                    print(f"✅ Found: {file_path.parent.name}/{file_path.name}")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    return documents

def process_document(document):
    """Chunks text using Llama 3."""
    prompt = f"Split this {document['type']} file into smart chunks: {document['text']}"
    try:
        response = llm_structured.invoke(prompt)
        return [{"page_content": f"{c.headline}\n{c.original_text}", 
                 "metadata": {"source": document["source"], "type": document["type"]}} 
                for c in response.chunks]
    except: return []

def build_database():
    """THE CORE ACTION: Reads files and creates the database."""
    print("🚀 Starting Database Build...")
    docs = fetch_documents()
    
    if not docs:
        print("🛑 No documents were found. Check your folder names!")
        return

    all_chunks = []
    for doc in docs:
        print(f"🧠 Processing {doc['source']} with Llama 3...")
        all_chunks.extend(process_document(doc))
    
    # Delete old DB if it exists
    if os.path.exists(DB_NAME):
        import shutil
        shutil.rmtree(DB_NAME)

    # Save to Chroma
    vectorstore = Chroma.from_texts(
        texts=[c["page_content"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
        embedding=embeddings,
        persist_directory=DB_NAME
    )
    print(f"🎉 SUCCESS! Database built with {len(all_chunks)} chunks.")

# ==========================================
# 4. CRITICAL: THIS CALLS THE FUNCTIONS!
# ==========================================
if __name__ == "__main__":
    build_database()
