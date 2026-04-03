import gradio as gr
from day5 import chat, db_status

theme = gr.themes.Soft()

with gr.Blocks(title="InsureLLM RAG System", theme=theme) as app:
    gr.Markdown("# 🏢 InsureLLM Knowledge Base Assistant")
    gr.Markdown("Ask questions about employees, products, or contracts.")
    
    # Show database status at the top
    gr.Markdown(f"**System Status:** {db_status}")

    doc_filter = gr.Dropdown(
        choices=["all", "employees", "products", "contracts"],
        value="all",
        label="Filter by document type",
        info="Narrow your search to a specific category"
    )

    def wrapped_chat(message, history, doc_filter):
        if not message.strip():
            return "Please type a question first."
        filter_value = None if doc_filter == "all" else doc_filter
        try:
            return chat(message, history, filter_value)
        except Exception as e:
            return f"An error occurred: {str(e)}. Please try again."

    gr.ChatInterface(
        fn=wrapped_chat,
        additional_inputs=[doc_filter],
        examples=[
            ["What insurance products does InsureLLM offer?"],
            ["Who are the employees in the sales department?"],
            ["What are the contract terms?"]
        ]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)