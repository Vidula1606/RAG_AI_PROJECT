import gradio as gr
from day5 import chat

theme = gr.themes.Soft()

with gr.Blocks(title="InsureLLM RAG System", theme=theme) as app:

    gr.Markdown("# InsureLLM Knowledge Base")
    gr.Markdown("Groq + Hybrid RAG + Memory")

    doc_filter = gr.Dropdown(
        choices=["all", "employees", "products", "contracts"],
        value="all",
        label="Filter by document type"
    )

    def wrapped_chat(message, history, doc_filter):
        filter_value = None if doc_filter == "all" else doc_filter
        return chat(message, history, filter_value)

    gr.ChatInterface(
        fn=wrapped_chat,
        additional_inputs=[doc_filter]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)