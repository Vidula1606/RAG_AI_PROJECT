import os
import glob
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# ==========================================
# Configuration
# ==========================================
DB_NAME = "vector_db"
KNOWLEDGE_BASE_DIR = "knowledge-base/"

# Use local HuggingFace embeddings (Free, no API keys needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_and_chunk_documents():
    """Loads markdown files from subfolders and splits them into chunks."""
    print("Loading documents from knowledge base...")
    folders = glob.glob(f"{KNOWLEDGE_BASE_DIR}*")
    
    documents = []
    for folder in folders:
        if os.path.isdir(folder):
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder, 
                glob="**/*.md", 
                loader_cls=TextLoader, 
                loader_kwargs={'encoding': 'utf-8'}
            )
            folder_docs = loader.load()
            
            # Tag each document with its folder name (e.g., 'employees', 'products')
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
                
    print(f"Loaded {len(documents)} documents.")

    # Split documents into smaller chunks for the vector database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Divided documents into {len(chunks)} chunks.")
    
    return chunks

def build_vector_database(chunks):
    """Embeds the chunks and stores them in a local Chroma database."""
    print("Building vector database. This may take a moment...")
    
    # Clear the old database if it exists to avoid duplicates
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    # Create and save the new database
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_NAME
    )
    
    collection = vectorstore._collection
    count = collection.count()
    print(f"Vectorstore successfully created with {count} documents!")
    return collection

def visualize_vectors(collection):
    """Generates a 3D visualization of the vectors using Plotly."""
    print("Generating 3D Visualization...")
    
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    metadatas = result['metadatas']
    
    doc_types = [metadata['doc_type'] for metadata in metadatas]
    
    # Map document types to colors
    color_map = {'products': 'blue', 'employees': 'green', 'contracts': 'red', 'company': 'orange'}
    colors = [color_map.get(t, 'gray') for t in doc_types]

    # Reduce dimensions to 3D for visualization
    tsne = TSNE(n_components=3, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        width=900,
        height=700,
        margin=dict(r=10, b=10, l=10, t=40)
    )

    fig.show()

if __name__ == "__main__":
    # 1. Process the files
    docs = load_and_chunk_documents()
    
    # 2. Build the database
    db_collection = build_vector_database(docs)
    
    # 3. Show the visualization in your browser
    visualize_vectors(db_collection)
