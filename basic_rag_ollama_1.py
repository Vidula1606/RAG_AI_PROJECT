import glob
from pathlib import Path
import gradio as gr
import ollama

# ==========================================
# 1. Load Knowledge Base (Simulated Vector Store)
# ==========================================
knowledge = {}

# Load employee data
employee_files = glob.glob("knowledge-base/employees/*")
for filename in employee_files:
    name = Path(filename).stem.split(" ")[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

# Load product data
product_files = glob.glob("knowledge-base/products/*")
for filename in product_files:
    name = Path(filename).stem
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

# ==========================================
# 2. Retrieval Logic (The "R" in RAG)
# ==========================================
SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""


def get_relevant_context(message):
    """Extracts keywords from the message and matches them to the knowledge base."""
    text = "".join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]


def additional_context(message):
    """Formats the retrieved context for the LLM prompt."""
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        return "There is no additional context relevant to the user's question."

    result = "The following additional context might be relevant in answering the user's question:\n\n"
    result += "\n\n".join(relevant_context)
    return result


# ==========================================
# 3. Generation Logic (The "G" in RAG)
# ==========================================
def chat(message, history):
    #  Fix message format
    if isinstance(message, list):
        message = message[0]["text"]

    system_message = SYSTEM_PREFIX + additional_context(message)

    messages = [{"role": "system", "content": system_message}]

    #  Fix history (new Gradio format)
    for msg in history:
        content = msg.get("content")

        # handle weird list format
        if isinstance(content, list):
            content = content[0]["text"]

        messages.append({"role": msg.get("role"), "content": content})

    # Add current message
    messages.append({"role": "user", "content": message})

    response = ollama.chat(model="llama3", messages=messages)

    return response["message"]["content"]


# ==========================================
# 4. Launch Application Interface
# ==========================================
if __name__ == "__main__":
    # Create and launch the Gradio UI
    demo = gr.ChatInterface(
        fn=chat,
        title="InsureLLM Knowledge Assistant",
        description="Ask me about our employees and products. Powered by local AI.",
    )
    demo.launch(inbrowser=True)
