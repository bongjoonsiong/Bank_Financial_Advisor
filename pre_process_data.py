# pre_process_data.py (Modified - Solution 1: Implicit Persist)
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

def main():
    """Pre-processes the 'Comprehensive Bank Information.pdf' document,
    creates embeddings using HuggingFaceEmbeddings, and builds a ChromaDB vector database (implicit persist).
    """

    pdf_file_path = "Comprehensive_Bank_Information.pdf"
    persist_directory = "chroma_db"

    # --- 1. Load Bank Product Information from PDF ---
    print(f"Loading document from: {pdf_file_path}")
    try:
        loader = PyPDFLoader(pdf_file_path)
        documents_pdf = loader.load()
        text_content = [doc.page_content for doc in documents_pdf]
        bank_product_info = text_content
        print(f"Successfully loaded {len(documents_pdf)} pages from PDF.")
    except FileNotFoundError:
        print(f"Error: PDF file not found at path: {pdf_file_path}. Please ensure the PDF file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    if not bank_product_info:
        print("No text content loaded from PDF. Exiting.")
        return

    # --- 2. Initialize HuggingFace Embeddings Model ---
    print("Initializing HuggingFace embeddings model...")
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # --- 3. Text Splitting ---
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents(bank_product_info)
    print(f"Split document into {len(documents)} chunks.")

    # --- 4. Create and Persist ChromaDB Vector Store (Implicit Persist) ---
    print(f"Creating and persisting ChromaDB vector store in: {persist_directory}...")
    # Initialize Chroma with persist_directory directly
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb.add_documents(documents=documents) # Add documents to the Chroma instance
    # vectordb.persist() # REMOVE this line - persist might be implicit now
    print(f"Vector database successfully created and (implicitly) persisted in '{persist_directory}' directory.")


if __name__ == "__main__":
    main()