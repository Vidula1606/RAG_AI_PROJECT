import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. Database Connection & Model Setup
# ==========================================
DB_NAME = "vector_db"

# Connect to the local Chroma database you built earlier
print("Connecting to vector database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# Turn the database into a LangChain Retriever
retriever = vectorstore.as_retriever()

# Initialize the local Llama model via Ollama
# (temperature=0 makes the model more factual and less prone to hallucination)
print("Initializing local LLM...")
llm = ChatOllama(model="llama3", temperature=0)

# ==========================================
# 2. Prompt & RAG Logic
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
    """Retrieves context and generates a response using the local LLM."""
    
    # Step A: Retrieve relevant documents from Chroma
    docs = retriever.invoke(question)
    
    # Step B: Format the retrieved text into a single block
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Step C: Inject the context into the system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    # Step D: Send the prompt and the user's question to the Llama model
    messages = [
        SystemMessage(content=system_prompt), 
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    
    return response.content

# ==========================================
# 3. Launch Application Interface
# ==========================================
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=answer_question,
        title="InsureLLM Assistant",
        description="A professional RAG chatbot powered by local Llama 3 and LangChain."
    )
    demo.launch(inbrowser=True)
