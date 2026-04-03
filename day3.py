import gradio as gr
import os
from dotenv import load_dotenv

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
# 2. Database Connection & Model Setup
# ==========================================
DB_NAME = "vector_db"

print("Connecting to vector database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

retriever = vectorstore.as_retriever()

# Initialize Groq LLM (fast inference)
print("Initializing Groq LLM...")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",  # or "mixtral-8x7b-32768"
    temperature=0
)

# ==========================================
# 3. Prompt & RAG Logic
# ==========================================
SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""

def answer_question(question: str, history):
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Combine retrieved content
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Create system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    # Prepare messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # Generate response
    response = llm.invoke(messages)
    
    return response.content

# ==========================================
# 4. Launch Application Interface
# ==========================================
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=answer_question,
        title="InsureLLM Assistant (Groq)",
        description="A professional RAG chatbot powered by Groq + LangChain."
    )
    demo.launch(inbrowser=True)