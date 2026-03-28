import gradio as gr

# Import chat function from day5
from day5 import chat

# ==========================================
# UI Setup
# ==========================================

theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

with gr.Blocks(title="InsureLLM RAG System", theme=theme) as app:

    gr.Markdown("# 🏢 InsureLLM Corporate Knowledge Base")
    gr.Markdown("Powered by Local Llama 3 + ChromaDB")

    # Only Chatbot (Removed evaluation tab)
    gr.ChatInterface(
        fn=chat,
        description="Ask me anything about InsureLLM's products, employees, or contracts.",
    )

# ==========================================
# Launch
# ==========================================

if __name__ == "__main__":
    print("Launching InsureLLM Chatbot...")
    app.launch(inbrowser=True)