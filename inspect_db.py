from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Connect to your database
BASE_DIR = Path(__file__).parent.resolve()
DB_NAME = str(BASE_DIR / "advanced_vector_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# Ask the database how many items it has
count = vectorstore._collection.count()
print(f"==========================================")
print(f"  DATABASE STATUS: {count} Chunks Found")
print(f"==========================================")

if count > 0:
    # Peek at the very first thing in the database
    sample = vectorstore._collection.get(limit=1)
    print(f"\nSAMPLE DATA IN DATABASE:")
    print(sample['documents'][0][:200] + "...")
else:
    print("\nTHE DATABASE IS EMPTY.")
    print("Check your 'knowledge-base' folder path in day5.py!")
