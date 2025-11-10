import os
import pandas as pd
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import logging
import hashlib
from typing import List, Tuple
import chromadb
from multiprocessing import Pool

os.environ["ANONYMIZED_TELEMETRY"] = "false"


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  
    CHUNK_SIZE = 2000 
    CHUNK_OVERLAP = 100  
    BATCH_SIZE = 1024 
    COLLECTION_NAME = "career_knowledge_base"

# Dynamic path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "datasets", "cleaned_data", "career_compass_master_datasets_1.csv"))
CHROMA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "db", "chroma", "csv_data"))
EMBEDDINGS_PICKLE_PATH = os.path.join(CHROMA_DIR, "embeddings.pkl")

def validate_paths() -> None:
    """Validate input and output paths."""
    if not os.path.exists(os.path.dirname(CSV_PATH)):
        logger.error(f"CSV directory does not exist: {os.path.dirname(CSV_PATH)}")
        raise FileNotFoundError("CSV directory not found")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    logger.info(f"Chroma directory ensured: {CHROMA_DIR}")

def load_and_validate_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if "rag_text" not in df.columns:
            logger.error("Required column 'rag_text' not found in CSV")
            raise ValueError("CSV missing 'rag_text' column")
        null_count = df["rag_text"].isna().sum()
        logger.info(f"Loaded CSV with {len(df)} rows, {null_count} null 'rag_text' entries")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        raise

def chunk_row(args):
    idx, row, splitter = args
    if pd.isna(row["rag_text"]) or not row["rag_text"].strip():
        return []
    chunks = splitter.split_text(row["rag_text"])
    return [
        Document(
            page_content=chunk,
            metadata={"id": str(row["id"]), "source": row.get("source", "unknown"), "chunk_id": i, "row_index": idx}
        )
        for i, chunk in enumerate(chunks)
    ]

def create_documents(df: pd.DataFrame, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Create document chunks from CSV data using multiprocessing."""
    documents = []
    skipped_rows = 0
    with Pool() as pool:
        results = pool.map(chunk_row, [(idx, row, splitter) for idx, row in df.iterrows()])
        for result in results:
            if result:
                documents.extend(result)
            else:
                skipped_rows += 1
    logger.info(f"Created {len(documents)} chunks, skipped {skipped_rows} rows")
    return documents

def compute_data_hash(documents: List[Document]) -> str:
    """Compute a hash of document content for validation."""
    content = "".join(doc.page_content for doc in documents)
    return hashlib.sha256(content.encode()).hexdigest()

def load_or_create_embeddings(documents: List[Document], embeddings: HuggingFaceEmbeddings) -> Tuple[List, bool]:
    """Load existing embeddings or create new ones with optimized batching."""
    data_hash = compute_data_hash(documents)
    pickle_path = EMBEDDINGS_PICKLE_PATH
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                saved_data = pickle.load(f)
                saved_embeddings, saved_hash = saved_data["embeddings"], saved_data["hash"]
                if saved_hash == data_hash and len(saved_embeddings) == len(documents):
                    logger.info("Loaded valid embeddings from pickle")
                    return saved_embeddings, True
                logger.warning("Embedding hash or count mismatch, regenerating embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embeddings pickle: {e}, regenerating embeddings")
    
    logger.info("Creating new embeddings")
    all_embeddings = []
    all_texts = [doc.page_content for doc in documents]
    for i in tqdm(range(0, len(all_texts), Config.BATCH_SIZE), desc="Embedding batches"):
        batch_texts = all_texts[i:i + Config.BATCH_SIZE]
        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error embedding batch {i // Config.BATCH_SIZE}: {e}")
            raise
    
    with open(pickle_path, "wb") as f:
        pickle.dump({"embeddings": all_embeddings, "hash": data_hash}, f)
    logger.info(f"Saved embeddings to {pickle_path}")
    return all_embeddings, False

def initialize_chroma(documents: List[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Initialize or update Chroma database without deleting existing data."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()

        if Config.COLLECTION_NAME not in [c.name for c in collections]:
            vector_db = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=Config.COLLECTION_NAME
            )
            logger.info(f"Created new Chroma collection: {Config.COLLECTION_NAME}")
        else:
            vector_db = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name=Config.COLLECTION_NAME
            )
            logger.info(f"Loaded existing Chroma collection: {Config.COLLECTION_NAME}")

            existing_count = vector_db._collection.count()
            vector_db.add_documents(documents)
            logger.info(f"Appended {len(documents)} new documents to collection "
                        f"'{Config.COLLECTION_NAME}' (previously had {existing_count})")

        return vector_db
    except Exception as e:
        logger.error(f"Failed to initialize or update Chroma DB: {e}")
        raise

def main():
    """Main function to ingest data and create embeddings."""
    validate_paths()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    
    df = load_and_validate_csv(CSV_PATH)
    
    documents = create_documents(df, splitter)
    if not documents:
        logger.error("No valid documents created, exiting")
        return
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.MODEL_NAME,
            model_kwargs={"device": "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"}
        )
        logger.info(f"Loaded embeddings model: {Config.MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {e}")
        raise
    
    _, loaded = load_or_create_embeddings(documents, embeddings)
    
    vector_db = initialize_chroma(documents, embeddings)
    
    count = vector_db._collection.count()
    logger.info(f"Ingestion complete! {count} chunks stored in Chroma DB at: {CHROMA_DIR}")
    if count != len(documents):
        logger.warning(f"Stored count ({count}) does not match created chunks ({len(documents)})")
    
    if documents:
        test_query = "career advice for software engineers"
        results = vector_db.similarity_search(test_query, k=3)
        for i, doc in enumerate(results):
            logger.info(f"Test query result {i+1}: {doc.page_content[:100]}... (Metadata: {doc.metadata})")

    try:
        if vector_db and hasattr(vector_db, "_client"):
            vector_db._client.persist()
            logger.info("✅ Chroma DB persisted successfully.")
        else:
            logger.warning("⚠️ No Chroma client found to persist.")
    except Exception as e:
        logger.error(f"Error persisting Chroma DB: {e}")

    vector_db = None
    logger.info(" Cleanup complete. Script finished.")

if __name__ == "__main__":
    import time 
    start = time.time()
    main()
    logger.info(f"Total runtime : {time.time()-start:.2f} seconds")
