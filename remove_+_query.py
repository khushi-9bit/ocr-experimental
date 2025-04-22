import os
import glob
import uuid
import logging
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")

# Discover all PDF files in the datasource folder
PDF_PATHS = glob.glob("datasource/*.pdf")

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings()
)

embedding_function = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def store_chunks_in_chromadb(pdf_paths):
    chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"Collection '{COLLECTION_NAME}' successfully initialized.")

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file not found at {pdf_path}")
            continue

        pdf_name = os.path.basename(pdf_path).split(".")[0]
        logger.info(f"Processing PDF: {pdf_name}")

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            logger.info(f"PDF split into {len(chunks)} chunks.")

            chunk_ids = [f"{pdf_name}_chunk_{uuid.uuid4().hex}" for _ in range(len(chunks))]
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = embedding_function.encode(chunk_texts, batch_size=16)
            metadata = [{"source": pdf_name} for _ in range(len(chunks))]

            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunk_texts
            )
            logger.info(f"Added {len(chunks)} chunks from {pdf_name} to the collection.")
        except Exception as e:
            logger.error(f"Error processing {pdf_name}: {e}")

    logger.info("All PDFs processed and chunks stored in ChromaDB.")
    return collection

if __name__ == "__main__":
    try:
        collection = store_chunks_in_chromadb(PDF_PATHS)
        logger.info(f"Collection response: {collection}")
    except Exception as e:
        logger.error(f"Error querying the database: {e}")
