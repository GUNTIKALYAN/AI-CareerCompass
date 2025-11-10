# import os
# import logging
# from typing import List, Union, Optional
# from datetime import datetime
# from dotenv import load_dotenv
# import chromadb

# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings


# # ----------------------------------------------------------------------
# # Logging & env
# # ----------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)
# load_dotenv()


# # ----------------------------------------------------------------------
# # Config
# # ----------------------------------------------------------------------
# class Config:
#     MODEL_NAME = "sentence-transformers/all-MPNet-base-v2"
#     CHUNK_SIZE = 800
#     CHUNK_OVERLAP = 100
#     COLLECTION_PREFIX = "user_session"


# # ----------------------------------------------------------------------
# # PATHS – **THIS IS THE ONLY CHANGE**
# # ----------------------------------------------------------------------
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# USER_SESSIONS_ROOT = os.path.join(BASE_DIR, "app", "db","chroma", "user_sessions")   # <-- NEW


# # ----------------------------------------------------------------------
# # Embeddings cache
# # ----------------------------------------------------------------------
# _embeddings = None

# def get_embeddings() -> HuggingFaceEmbeddings:
#     global _embeddings
#     if _embeddings is None:
#         try:
#             _embeddings = HuggingFaceEmbeddings(
#                 model_name=Config.MODEL_NAME,
#                 model_kwargs={"device": "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"}
#             )
#             logger.info(f"Loaded embeddings model: {Config.MODEL_NAME}")
#         except Exception as e:
#             logger.error(f"Failed to load embeddings model: {e}")
#             raise
#     return _embeddings


# # ----------------------------------------------------------------------
# # Validation & directory setup
# # ----------------------------------------------------------------------
# def validate_user_id(user_id: str) -> None:
#     if not user_id or any(c in user_id for c in ["/", "\\", ".."]):
#         logger.error(f"Invalid user_id: {user_id}")
#         raise ValueError("Invalid user_id")


# def setup_directories(user_id: str) -> str:
#     """Create user-specific Chroma directory under app/db/user_sessions/<user_id>."""
#     validate_user_id(user_id)
#     user_chroma_dir = os.path.join(USER_SESSIONS_ROOT, user_id)
#     os.makedirs(user_chroma_dir, exist_ok=True)
#     logger.info(f"Ensured directory: {user_chroma_dir}")
#     return user_chroma_dir


# # ----------------------------------------------------------------------
# # Document creation (unchanged)
# # ----------------------------------------------------------------------
# def create_documents(user_id: str, answers: Union[str, List[str]], answer_id: Optional[str] = None) -> List[Document]:
#     answers = [answers] if isinstance(answers, str) else answers
#     valid_answers = [str(a).strip() for a in answers if str(a).strip()]
#     if not valid_answers:
#         logger.warning(f"No valid answers provided for user_id: {user_id}")
#         return []

#     session_text = " ".join(valid_answers)
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=Config.CHUNK_SIZE,
#         chunk_overlap=Config.CHUNK_OVERLAP,
#         length_function=len,
#         add_start_index=True
#     )
#     chunks = splitter.split_text(session_text)
#     documents = [
#         Document(
#             page_content=chunk,
#             metadata={
#                 "user_id": user_id,
#                 "session_id": user_id,
#                 "chunk_id": i,
#                 "source": "user_answers",
#                 "answer_id": answer_id or f"answer_{datetime.now().isoformat()}",
#                 "timestamp": datetime.now().isoformat()
#             }
#         )
#         for i, chunk in enumerate(chunks)
#     ]
#     logger.info(f"Created {len(documents)} chunks for user_id: {user_id}")
#     return documents


# # ----------------------------------------------------------------------
# # Chroma helper (unchanged)
# # ----------------------------------------------------------------------
# def get_or_create_chroma(user_id: str, user_chroma_dir: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
#     try:
#         collection_name = f"{Config.COLLECTION_PREFIX}_{user_id}"
#         vector_db = Chroma(
#             collection_name=collection_name,
#             embedding_function=embeddings,
#             persist_directory=user_chroma_dir
#         )
#         logger.info(f"Connected to Chroma collection: {collection_name}")
#         return vector_db
#     except Exception as e:
#         logger.error(f"Failed to initialize Chroma for user_id {user_id}: {e}")
#         raise


# # ----------------------------------------------------------------------
# # Public ingest function (unchanged – it now receives the new path)
# # ----------------------------------------------------------------------
# def ingest_user_answers(user_id: str, answers: Union[str, List[str]], answer_id: Optional[str] = None) -> bool:
#     try:
#         user_chroma_dir = setup_directories(user_id)          # <-- NEW PATH
#         embeddings = get_embeddings()
#         documents = create_documents(user_id, answers, answer_id)
#         if not documents:
#             logger.warning(f"No documents created for user_id: {user_id}")
#             return False

#         vector_db = get_or_create_chroma(user_id, user_chroma_dir, embeddings)
#         vector_db.add_documents(documents)

#         count = vector_db._collection.count()
#         logger.info(f"Added {len(documents)} chunks for user_id {user_id}. Total chunks: {count}")

#         # optional test query
#         if documents:
#             test_query = "user preferences"
#             results = vector_db.similarity_search(test_query, k=3)
#             for i, doc in enumerate(results):
#                 logger.info(f"Test result {i+1}: {doc.page_content[:100]}... (Metadata: {doc.metadata})")

#         return True
#     except Exception as e:
#         logger.error(f"Failed to ingest answers for user_id {user_id}: {e}")
#         return False


# # ----------------------------------------------------------------------
# # Demo
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     test_user_id = "507f1f77bcf86cd799439011"
#     test_answers = [
#         "I am interested in software engineering and want to learn Python.",
#         "I have 2 years of experience in web development."
#     ]
#     success = ingest_user_answers(test_user_id, test_answers, answer_id="test_001")
#     logger.info(f"Ingestion {'successful' if success else 'failed'} for user_id: {test_user_id}")



# app/scripts/ingest_user_data.py
import os
import logging
from typing import List, Union, Optional
from datetime import datetime
from dotenv import load_dotenv

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Logging & env
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

# Config
class Config:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    COLLECTION_PREFIX = "user_session"



# PATHS – user-specific Chroma directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USER_SESSIONS_ROOT = os.path.join(BASE_DIR, "app", "db", "chroma", "user_sessions")


# Embeddings cache
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name=Config.MODEL_NAME,
                model_kwargs={"device": "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"},
            )
            logger.info(f"Loaded embeddings model: {Config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise
    return _embeddings


# Validation & directory setup
def validate_user_id(user_id: str) -> None:
    if not user_id or any(c in user_id for c in ("/", "\\", "..")):
        logger.error(f"Invalid user_id: {user_id}")
        raise ValueError("Invalid user_id")


def setup_directories(user_id: str) -> str:
    """Create (or ensure) a Chroma directory for the user."""
    validate_user_id(user_id)
    user_chroma_dir = os.path.join(USER_SESSIONS_ROOT, user_id)
    os.makedirs(user_chroma_dir, exist_ok=True)
    logger.info(f"Ensured directory: {user_chroma_dir}")
    return user_chroma_dir


# Document creation helpers
def _split_text(text: str) -> List[str]:
    """Split a long answer into chunks (preserves context)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_text(text)


def create_documents_from_answers(
    user_id: str,
    answers: Union[str, List[str]],
    answer_id: Optional[str] = None,
) -> List[Document]:
    """
    Convert one or many raw answers into LangChain ``Document`` objects.
    Each answer becomes **one** Document (splitting only if > CHUNK_SIZE).
    """
    if isinstance(answers, str):
        answers = [answers]

    docs: List[Document] = []
    for idx, raw in enumerate(answers):
        raw = raw.strip()
        if not raw:
            continue

        # split only if needed
        chunks = _split_text(raw) if len(raw) > Config.CHUNK_SIZE else [raw]

        for chunk_idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "user_id": user_id,
                    "session_id": user_id,               # same as user_id for simplicity
                    "answer_index": idx,
                    "chunk_index": chunk_idx,
                    "source": "user_answers",
                    "answer_id": answer_id or f"answer_{datetime.now().isoformat()}",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            docs.append(doc)

    logger.info(f"Created {len(docs)} document(s) for user_id: {user_id}")
    return docs


# Chroma helper
def get_or_create_chroma(
    user_id: str, user_chroma_dir: str, embeddings: HuggingFaceEmbeddings
) -> Chroma:
    try:
        collection_name = f"{Config.COLLECTION_PREFIX}_{user_id}"
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=user_chroma_dir,
        )
        logger.info(f"Connected to Chroma collection: {collection_name}")
        return vector_db
    except Exception as e:
        logger.error(f"Failed to initialise Chroma for user_id {user_id}: {e}")
        raise


# Public API – ingest a *single* answer (used by the Flask route)
def ingest_user_answers(
    user_id: str,
    answers: Union[str, List[str]],
    answer_id: Optional[str] = None,
) -> bool:
    """
    Called from ``career_bp.chat`` for every new user message.
    Returns ``True`` on success.
    """
    try:
        user_chroma_dir = setup_directories(user_id)
        embeddings = get_embeddings()
        docs = create_documents_from_answers(user_id, answers, answer_id)

        if not docs:
            logger.warning(f"No documents created for user_id: {user_id}")
            return False

        vector_db = get_or_create_chroma(user_id, user_chroma_dir, embeddings)
        vector_db.add_documents(docs)

        total = vector_db._collection.count()
        logger.info(
            f"Added {len(docs)} chunk(s) for user_id {user_id}. Total docs now: {total}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to ingest answers for user_id {user_id}: {e}")
        return False


# NEW – ingest **all** historic answers for a user (back-fill)
def ingest_historic_user_answers(user_id: str) -> bool:
    """
    Pulls the complete chat history from MongoDB (via db_utils) and
    stores **every user message** in the user-specific Chroma DB.
    Useful for initial population or re-indexing.
    """
    try:
        from app.utils.db_utils import get_career_chat

        chat_session = get_career_chat(user_id)
        if not chat_session or not chat_session.get("messages"):
            logger.info(f"No chat history found for user_id: {user_id}")
            return True

        # Extract only the user messages (ignore AI responses)
        user_messages = [
            msg["user_message"]
            for msg in chat_session["messages"]
            if msg.get("user_message")
        ]

        if not user_messages:
            logger.info(f"No user answers in history for user_id: {user_id}")
            return True

        # Use a generic answer_id for the whole batch
        answer_id = f"historic_batch_{datetime.now().isoformat()}"
        return ingest_user_answers(user_id, user_messages, answer_id)

    except Exception as e:
        logger.error(f"Error ingesting historic data for user_id {user_id}: {e}")
        return False



if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest user answers into per-user Chroma DB."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("user_id", nargs="?", help="Single user_id to index")
    group.add_argument(
        "--all", action="store_true", help="Index **all** users that have chat history"
    )
    args = parser.parse_args()

    # 10 SAMPLE USER INSIGHTS (covering all question groups)
    SAMPLE_ANSWERS = [
        # academics
        "I'm in 10th grade, CBSE. I love Maths and Science — I scored 98 in Maths in pre-boards. "
        "English is okay, but I hate memorizing poems and grammar rules.",

        # family_context
        "My dad is a software engineer at Infosys, mom is a homemaker. "
        "They want me to become a doctor or engineer — no pressure for family business.",

        # interests
        "I like learning new things on my computer."
        "I’ve tried basic coding in Python"

        # aspirations
        "I want to become an engineer in the future, maybe in computer or electronics field"
        "I value creative freedom and the chance to solve real-world problems over just a high salary.",

        # preferences
        "I want to study in a good college in India. "
        "I’m fine with 4-5 years of formal education.",

        # personality
        "I’m an ambivert – I enjoy leading a small team but also need quiet time to think through complex algorithms. "
        "Deadlines energise me as long as the goals are clear."
    ]

    if args.all:
        from pymongo import MongoClient

        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
        db = client["Ai_career_roadmap"]
        chat_coll = db.chat_histories

        user_ids = [
            str(doc["user_id"])
            for doc in chat_coll.find({}, {"user_id": 1})
            if "user_id" in doc
        ]
        logger.info(f"Found {len(user_ids)} users with chat history. Starting back-fill…")

        success_cnt = 0
        for uid in user_ids:
            if ingest_historic_user_answers(uid):
                success_cnt += 1
            else:
                logger.warning(f"Failed for user_id {uid}")

        logger.info(f"Back-fill finished: {success_cnt}/{len(user_ids)} successful.")

    else:
        # SINGLE-USER DEMO – now with 10 insightful answers
        test_user_id = args.user_id or "6907852acce533e5a71cc9e3"
        ok = ingest_user_answers(
            user_id=test_user_id,
            answers=SAMPLE_ANSWERS,
            answer_id="sample_insights_001"
        )
        logger.info(
            f"Demo ingestion {'successful' if ok else 'failed'} for user_id: {test_user_id}"
        )



