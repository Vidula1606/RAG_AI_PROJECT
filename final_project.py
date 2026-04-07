import gradio as gr
from advanced_retrieval_logic_4 import chat, DB_NAME
import os

# ==========================================
# 1. Helper Functions
# ==========================================
def get_status():
    if os.path.exists(DB_NAME):
        return "✅ System Ready"
    return "❌ Database Missing"

def wrapped_chat(message, history, doc_filter):
    filter_val = None if doc_filter == "all" else doc_filter
    return chat(message, history, filter_val)

# ==========================================
# 2. UI Layout
# ==========================================
with gr.Blocks(title="InsureLLM RAG System", theme=gr.themes.Default()) as app:
    gr.Markdown("# 🏢 InsureLLM Knowledge Assistant")
    status_label = gr.Markdown(f"**Status:** {get_status()}")

    doc_filter = gr.Dropdown(
        choices=["all", "employees", "products", "contracts"],
        value="all",
        label="Search Category"
    )

    gr.ChatInterface(
        fn=wrapped_chat,
        additional_inputs=[doc_filter],
        fill_height=False,
        examples=[
            ["What insurance products does InsureLLM offer?"],
            ["Who are the employees in the sales department?"],
            ["What are the contract terms?"]
        ]    )

# ==========================================
# 3. Launch
# ==========================================
if __name__ == "__main__":
    app.launch(inbrowser=True)