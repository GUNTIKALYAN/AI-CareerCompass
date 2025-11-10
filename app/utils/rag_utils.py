# # # #!/usr/bin/env python3
# # # """
# # # rag_utils.py

# # # Retrieves user answers from Chroma and ONET data, analyzes,
# # # and produces 3–5 personalized career paths for 10th-grade Indian students.
# # # """

# # # import os
# # # import logging
# # # from pathlib import Path
# # # from typing import List, Dict, Optional
# # # from dotenv import load_dotenv
# # # import warnings

# # # # ------------------------------
# # # # LangChain + Chroma
# # # # ------------------------------
# # # from langchain_huggingface import HuggingFaceEmbeddings
# # # from langchain_chroma import Chroma
# # # import chromadb
# # # from groq import Groq

# # # # Suppress warnings
# # # warnings.filterwarnings("ignore", category=UserWarning)

# # # # ------------------------------
# # # # Logging
# # # # ------------------------------
# # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# # # logger = logging.getLogger(__name__)

# # # # ------------------------------
# # # # Load ENV + GROQ
# # # # ------------------------------
# # # load_dotenv()
# # # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# # # if not GROQ_API_KEY:
# # #     raise ValueError("GROQ_API_KEY not found in .env")
# # # groq_client = Groq(api_key=GROQ_API_KEY)

# # # # ------------------------------
# # # # Paths
# # # # ------------------------------
# # # PROJECT_ROOT = Path(__file__).resolve().parents[2]
# # # CHROMA_ROOT = PROJECT_ROOT / "app" / "db" / "chroma"
# # # USER_SESSIONS_ROOT = CHROMA_ROOT / "user_sessions"
# # # ONET_DB_PATH = CHROMA_ROOT / "csv_data"

# # # # ------------------------------
# # # # Config
# # # # ------------------------------
# # # class Config:
# # #     EMBEDDING_MODEL = "sentence-transformers/all-MPNet-base-v2"
# # #     TOP_K_SIMILAR = 10
# # #     FINAL_TOP_K = 5
# # #     GROQ_MODEL = "llama-3.1-8b-instant"

# # # # ------------------------------
# # # # Embeddings (cached)
# # # # ------------------------------
# # # _embeddings = None
# # # def get_embeddings() -> HuggingFaceEmbeddings:
# # #     global _embeddings
# # #     if _embeddings is None:
# # #         device = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
# # #         _embeddings = HuggingFaceEmbeddings(
# # #             model_name=Config.EMBEDDING_MODEL,
# # #             model_kwargs={"device": device}
# # #         )
# # #         logger.info(f"Embeddings loaded: {Config.EMBEDDING_MODEL} [{device}]")
# # #     return _embeddings

# # # # ------------------------------
# # # # Find user Chroma directory
# # # # ------------------------------
# # # def find_user_chroma_dir(user_id: str) -> Optional[Path]:
# # #     user_dir = USER_SESSIONS_ROOT / user_id
# # #     if not user_dir.exists():
# # #         logger.warning(f"User directory not found: {user_dir}")
# # #         return None
# # #     sqlite3_path = user_dir / "chroma.sqlite3"
# # #     if sqlite3_path.exists():
# # #         logger.info(f"Found Chroma DB: {sqlite3_path}")
# # #         return user_dir
# # #     logger.warning(f"chroma.sqlite3 not found in {user_dir}")
# # #     return None

# # # # ------------------------------
# # # # Load user DB
# # # # ------------------------------
# # # def get_user_db(user_id: str) -> Optional[Chroma]:
# # #     chroma_dir = find_user_chroma_dir(user_id)
# # #     if not chroma_dir:
# # #         return None
# # #     embed_fn = get_embeddings()
# # #     persist_dir = str(chroma_dir)
# # #     try:
# # #         client = chromadb.PersistentClient(path=persist_dir)
# # #         collections = client.list_collections()
# # #     except Exception as e:
# # #         logger.error(f"Failed to connect to Chroma client: {e}")
# # #         return None
# # #     if not collections:
# # #         logger.warning("No collections found in DB.")
# # #         return None
# # #     for coll in collections:
# # #         coll_name = coll.name
# # #         try:
# # #             db = Chroma(
# # #                 collection_name=coll_name,
# # #                 embedding_function=embed_fn,
# # #                 persist_directory=persist_dir
# # #             )
# # #             count = db._collection.count()
# # #             if count > 0:
# # #                 logger.info(f"Using collection '{coll_name}' ({count} docs)")
# # #                 return db
# # #         except Exception as e:
# # #             logger.debug(f"Failed to load '{coll_name}': {e}")
# # #     logger.warning("No non-empty collection found.")
# # #     return None

# # # # ------------------------------
# # # # Load ONET DB
# # # # ------------------------------
# # # def get_onet_db() -> Optional[Chroma]:
# # #     if not ONET_DB_PATH.exists():
# # #         logger.error(f"ONET DB path not found: {ONET_DB_PATH}")
# # #         return None
# # #     try:
# # #         db = Chroma(
# # #             collection_name="career_knowledge_base",  # <- correct collection
# # #             embedding_function=get_embeddings(),
# # #             persist_directory=str(ONET_DB_PATH)
# # #         )
# # #         logger.info(f"ONET DB loaded: {db._collection.count()} careers")
# # #         return db
# # #     except Exception as e:
# # #         logger.error(f"Failed to load ONET DB: {e}")
# # #         return None

# # # # ------------------------------
# # # # Extract user insights
# # # # ------------------------------
# # # def extract_user_insights(db: Chroma) -> str:
# # #     try:
# # #         docs = db.similarity_search("", k=500)
# # #         parts = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
# # #         if not parts:
# # #             return "No answers recorded."
# # #         logger.info(f"Extracted insights ({len(parts)} messages): {parts[:5]}...")
# # #         return " | ".join(parts)
# # #     except Exception as e:
# # #         logger.error(f"Insight extraction failed: {e}")
# # #         return "Error reading user data."

# # # # ------------------------------
# # # # Match with ONET
# # # # ------------------------------
# # # def match_with_onet(insights: str, onet_db: Chroma) -> List[Dict]:
# # #     try:
# # #         results = onet_db.similarity_search_with_score(insights, k=Config.TOP_K_SIMILAR)
# # #         matches = [
# # #             {
# # #                 "title": doc.metadata.get("title", "Unknown Career"),
# # #                 "desc": doc.page_content,
# # #                 "sim": round(1 - score, 3)
# # #             }
# # #             for doc, score in results
# # #         ]
# # #         logger.info(f"ONET raw matches: {len(matches)}")
# # #         return matches
# # #     except Exception as e:
# # #         logger.error(f"ONET similarity search failed: {e}")
# # #         return []

# # # # ------------------------------
# # # # Parse GROQ response
# # # # ------------------------------
# # # def parse_groq_response(text: str) -> List[Dict]:
# # #     matches = []
# # #     cur = {}
# # #     for line in text.split("\n"):
# # #         line = line.strip()
# # #         if line.startswith("Title:"):
# # #             if cur: matches.append(cur)
# # #             cur = {"title": line[6:].strip()}
# # #         elif line.startswith("Score:"):
# # #             try: cur["match_score"] = int(line.split("%")[0].split()[-1])
# # #             except: cur["match_score"] = 50
# # #         elif line.startswith("Rationale:"):
# # #             cur["rationale"] = line[10:].strip()
# # #         elif line.startswith("Roadmap:"):
# # #             cur["roadmap"] = line[8:].strip()
# # #         elif line.startswith("Timeline:"):
# # #             cur["timeline"] = line[9:].strip()
# # #     if cur: matches.append(cur)
# # #     return matches[:Config.FINAL_TOP_K]

# # # # ------------------------------
# # # # GROQ refinement
# # # # ------------------------------
# # # def match_with_groq(insights: str, matches: List[Dict]) -> List[Dict]:
# # #     if not matches:
# # #         return fallback(matches)
# # #     career_lines = [f"- {m['title']}: {m['desc'][:300]}..." for m in matches]
# # #     prompt = f"""
# # # You are CareerCompass AI for 10th-grade students in India.

# # # User Answers:
# # # {insights}

# # # Top ONET Careers:
# # # {chr(10).join(career_lines)}

# # # Return exactly 3–5 careers with:

# # # Title: <career name>
# # # Score: <0-100>%
# # # Rationale: <1-2 sentences>
# # # Roadmap: <11th-12th stream, entrance exams, degree>
# # # Timeline: <2026-2033+>
# # # """
# # #     try:
# # #         resp = groq_client.chat.completions.create(
# # #             model=Config.GROQ_MODEL,
# # #             messages=[{"role": "user", "content": prompt}],
# # #             max_tokens=1200,
# # #             temperature=0.7
# # #         )
# # #         return parse_groq_response(resp.choices[0].message.content.strip())
# # #     except Exception as e:
# # #         logger.error(f"GROQ error: {e}")
# # #         return fallback(matches)

# # # # ------------------------------
# # # # Fallback careers
# # # # ------------------------------
# # # def fallback(matches: List[Dict]) -> List[Dict]:
# # #     fallback_list = [
# # #         {
# # #             "title": m.get("title", "Unknown") if matches else "Software Engineer",
# # #             "match_score": max(30, int(m.get("sim", 0) * 100) if matches else 50),
# # #             "rationale": "Based on your answers.",
# # #             "roadmap": "11th-12th: MPC; Exam: JEE Main; B.Tech",
# # #             "timeline": "2026-2028: Prep, 2029-2032: Degree, 2033+: Job"
# # #         }
# # #         for m in matches[:3] if matches
# # #     ]
# # #     if not fallback_list:
# # #         fallback_list = [{
# # #             "title": "Software Engineer",
# # #             "match_score": 50,
# # #             "rationale": "Fallback career recommendation.",
# # #             "roadmap": "11th-12th: MPC; Exam: JEE Main; B.Tech",
# # #             "timeline": "2026-2028: Prep, 2029-2032: Degree, 2033+: Job"
# # #         }]
# # #     return fallback_list

# # # # ------------------------------
# # # # Main function
# # # # ------------------------------
# # # def match_careers(user_id: str) -> List[Dict]:
# # #     user_db = get_user_db(user_id)
# # #     if not user_db:
# # #         return [{"error": "No user data. Answer questions first."}]

# # #     onet_db = get_onet_db()
# # #     if not onet_db:
# # #         return [{"error": "Career database not ready. Contact admin."}]

# # #     insights = extract_user_insights(user_db)
# # #     if "No answers" in insights or "Error" in insights:
# # #         return [{"error": "No answers recorded."}]

# # #     onet_matches = match_with_onet(insights, onet_db)
# # #     return match_with_groq(insights, onet_matches)

# # # # ------------------------------
# # # # Test
# # # # ------------------------------
# # # def test():
# # #     user_id = "6901056d378d78bbadee1944"
# # #     print(f"\nCareer Matches for: {user_id}\n" + "="*70)
# # #     results = match_careers(user_id)
# # #     if not results:
# # #         print("No careers found.")
# # #     elif "error" in results[0]:
# # #         print(f"ERROR: {results[0]['error']}")
# # #     else:
# # #         for m in results:
# # #             print(f"{m.get('title', 'N/A')}")
# # #             print(f"   Score: {m.get('match_score', 'N/A')}%")
# # #             print(f"   Why: {m.get('rationale', 'N/A')}")
# # #             print(f"   Plan: {m.get('roadmap', 'N/A')}")
# # #             print(f"   When: {m.get('timeline', 'N/A')}\n")

# # # if __name__ == "__main__":
# # #     test()



# # #!/usr/bin/env python3
# # """
# # rag_utils.py – Corrected version
# # Ensures embedding dimensions match and user insights are properly used.
# # """

# # import os
# # import logging
# # from pathlib import Path
# # from typing import List, Dict, Optional
# # from dotenv import load_dotenv

# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_chroma import Chroma
# # import chromadb

# # from groq import Groq
# # import warnings
# # warnings.filterwarnings("ignore", category=UserWarning)

# # # ---------------- Logging ----------------
# # logging.basicConfig(level=logging.INFO,
# #                     format="%(asctime)s - %(levelname)s - %(message)s")
# # logger = logging.getLogger(__name__)

# # # ---------------- Env & GROQ ----------------
# # load_dotenv()
# # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# # if not GROQ_API_KEY:
# #     raise ValueError("GROQ_API_KEY not found in .env")
# # groq_client = Groq(api_key=GROQ_API_KEY)

# # # ---------------- Paths ----------------
# # PROJECT_ROOT = Path(__file__).resolve().parents[2]
# # CHROMA_ROOT = PROJECT_ROOT / "app" / "db" / "chroma"
# # USER_SESSIONS_ROOT = CHROMA_ROOT / "user_sessions"
# # ONET_DB_PATH = CHROMA_ROOT / "csv_data"

# # # ---------------- Config ----------------
# # class Config:
# #     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# #     TOP_K_SIMILAR = 10
# #     FINAL_TOP_K = 5
# #     GROQ_MODEL = "llama-3.1-8b-instant"

# # # ---------------- Embeddings ----------------
# # _embeddings = None
# # def get_embeddings() -> HuggingFaceEmbeddings:
# #     global _embeddings
# #     if _embeddings is None:
# #         device = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
# #         _embeddings = HuggingFaceEmbeddings(
# #             model_name=Config.EMBEDDING_MODEL,
# #             model_kwargs={"device": device}
# #         )
# #         logger.info(f"Embeddings loaded: {Config.EMBEDDING_MODEL} [{device}]")
# #     return _embeddings

# # # ---------------- User Chroma ----------------
# # def find_user_chroma_dir(user_id: str) -> Optional[Path]:
# #     user_dir = USER_SESSIONS_ROOT / user_id
# #     if not user_dir.exists():
# #         logger.warning(f"User directory not found: {user_dir}")
# #         return None
# #     sqlite3_path = user_dir / "chroma.sqlite3"
# #     if sqlite3_path.exists():
# #         logger.info(f"Found Chroma DB: {sqlite3_path}")
# #         return user_dir
# #     logger.warning(f"chroma.sqlite3 not found in {user_dir}")
# #     return None

# # def get_user_db(user_id: str) -> Optional[Chroma]:
# #     chroma_dir = find_user_chroma_dir(user_id)
# #     if not chroma_dir:
# #         return None

# #     embed_fn = get_embeddings()
# #     persist_dir = str(chroma_dir)

# #     try:
# #         client = chromadb.PersistentClient(path=persist_dir)
# #         collections = client.list_collections()
# #     except Exception as e:
# #         logger.error(f"Failed to connect to Chroma client: {e}")
# #         return None

# #     if not collections:
# #         logger.warning("No collections found in DB.")
# #         return None

# #     for coll in collections:
# #         coll_name = coll.name
# #         try:
# #             db = Chroma(
# #                 collection_name=coll_name,
# #                 embedding_function=embed_fn,
# #                 persist_directory=persist_dir
# #             )
# #             count = db._collection.count()
# #             if count > 0:
# #                 logger.info(f"Using collection '{coll_name}' ({count} docs)")
# #                 return db
# #         except Exception as e:
# #             logger.debug(f"Failed to load '{coll_name}': {e}")
# #     logger.warning("No non-empty collection found.")
# #     return None

# # # ---------------- ONET DB ----------------
# # # ----------------------------------------------------------------------
# # # Load ONET DB (FIXED COLLECTION)
# # # ----------------------------------------------------------------------
# # def get_onet_db() -> Optional[Chroma]:
# #     if not ONET_DB_PATH.exists():
# #         logger.error(f"ONET DB not found: {ONET_DB_PATH}")
# #         return None

# #     try:
# #         db = Chroma(
# #             collection_name="career_knowledge_base",  # ← NOT "csv_data"
# #             embedding_function=get_embeddings(),
# #             persist_directory=str(ONET_DB_PATH)
# #         )
# #         count = db._collection.count()
# #         if count == 0:
# #             logger.error("ONET DB is empty!")
# #             return None
# #         logger.info(f"ONET DB loaded: {count} careers")
# #         return db
# #     except Exception as e:
# #         logger.error(f"ONET load failed: {e}")
# #         return None
    
# # # ---------------- Extract insights ----------------
# # def extract_user_insights(db: Chroma) -> str:
# #     try:
# #         docs = db.similarity_search("", k=500)
# #         parts = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
# #         logger.info(f"Extracted insights ({len(parts)} messages): {parts[:5]}...")
# #         return " | ".join(parts) if parts else "No answers recorded."
# #     except Exception as e:
# #         logger.error(f"Insight extraction failed: {e}")
# #         return "Error reading user data."

# # # ---------------- GROQ Matching ----------------
# # def match_with_groq(insights: str, matches: List[Dict]) -> List[Dict]:
# #     if not matches:
# #         return fallback([])

# #     career_lines = [f"- {m['title']}: {m['desc'][:300]}..." for m in matches]
# #     prompt = f"""
# # You are CareerCompass AI for 10th-grade students in India.

# # User Answers:
# # {insights}

# # Top ONET Careers:
# # {chr(10).join(career_lines)}

# # Return exactly in JSON with at least 3 careers with:
# # Title, Score(0-100), Rationale, Roadmap, Timeline
# # Include Indian exams where relevant.
# # """

# #     try:
# #         resp = groq_client.chat.completions.create(
# #             model=Config.GROQ_MODEL,
# #             messages=[{"role": "user", "content": prompt}],
# #             max_tokens=1200,
# #             temperature=0.7
# #         )
# #         return parse_groq_response(resp.choices[0].message.content.strip())
# #     except Exception as e:
# #         logger.error(f"GROQ error: {e}")
# #         return fallback(matches)

# # def parse_groq_response(text: str) -> List[Dict]:
# #     import json
# #     try:
# #         data = json.loads(text)
# #         return data[:Config.FINAL_TOP_K]
# #     except Exception:
# #         # fallback parsing if JSON fails
# #         matches = []
# #         cur = {}
# #         for line in text.split("\n"):
# #             line = line.strip()
# #             if line.startswith("Title:"):
# #                 if cur: matches.append(cur)
# #                 cur = {"title": line[6:].strip()}
# #             elif line.startswith("Score:"):
# #                 try: cur["match_score"] = int(line.split("%")[0].split()[-1])
# #                 except: cur["match_score"] = 50
# #             elif line.startswith("Rationale:"):
# #                 cur["rationale"] = line[10:].strip()
# #             elif line.startswith("Roadmap:"):
# #                 cur["roadmap"] = line[8:].strip()
# #             elif line.startswith("Timeline:"):
# #                 cur["timeline"] = line[9:].strip()
# #         if cur: matches.append(cur)
# #         return matches[:Config.FINAL_TOP_K]

# # # ---------------- Fallback ----------------
# # def fallback(matches: List[Dict]) -> List[Dict]:
# #     return [
# #         {
# #             "title": m.get("title", "Software Engineer"),
# #             "match_score": max(30, int(m.get("sim", 0) * 100)),
# #             "rationale": "Fallback career recommendation.",
# #             "roadmap": "11th-12th: MPC; Exam: JEE Main; B.Tech",
# #             "timeline": "2026-2028: Prep, 2029-2032: Degree, 2033+: Job"
# #         }
# #         for m in (matches or [{}])[:3]
# #     ]

# # # ---------------- Main career matcher ----------------
# # def match_careers(user_id: str) -> List[Dict]:
# #     user_db = get_user_db(user_id)
# #     if not user_db:
# #         return [{"error": "No user data. Answer questions first."}]

# #     onet_db = get_onet_db()
# #     if not onet_db:
# #         return [{"error": "Career database not ready. Contact admin."}]

# #     insights = extract_user_insights(user_db)
# #     if "No answers" in insights:
# #         return [{"error": "No answers recorded."}]

# #     try:
# #         results = onet_db.similarity_search_with_score(insights, k=Config.TOP_K_SIMILAR)
# #         onet_matches = [
# #             {
# #                 "title": doc.metadata.get("title", "Unknown Career"),
# #                 "desc": doc.page_content,
# #                 "sim": round(1 - score, 3)
# #             }
# #             for doc, score in results
# #         ]
# #         logger.info(f"ONET raw matches: {len(onet_matches)}")
# #     except Exception as e:
# #         logger.error(f"ONET similarity search failed: {e}")
# #         onet_matches = []

# #     return match_with_groq(insights, onet_matches) or fallback(onet_matches)

# # # ---------------- Test ----------------
# # def test():
# #     user_id = "6901056d378d78bbadee1944"
# #     print(f"\nCareer Matches for: {user_id}\n" + "="*70)
# #     results = match_careers(user_id)
# #     if results and "error" in results[0]:
# #         print(f"ERROR: {results[0]['error']}")
# #     else:
# #         for m in results:
# #             print(f"{m['title']}")
# #             print(f"   Score: {m['match_score']}%")
# #             print(f"   Why: {m['rationale']}")
# #             print(f"   Plan: {m['roadmap']}")
# #             print(f"   When: {m['timeline']}\n")

# # if __name__ == "__main__":
# #     test()















# #!/usr/bin/env python3
# """
# rag_utils.py – RAG pipeline for career matching

# 1. Load user answers from per-user Chroma DB (created by ingest_user_data.py)
# 2. Retrieve top ONET careers via similarity search
# 3. Refine + personalize with Groq (JSON output)
# 4. Return 3–5 Indian-context career paths for 10th-grade students
# """

# import os
# import logging
# from pathlib import Path
# from typing import List, Dict, Optional
# from dotenv import load_dotenv
# import warnings

# # LangChain + Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# import chromadb

# # Groq
# from groq import Groq

# # Suppress noisy warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# # ------------------------------
# # Logging
# # ------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ------------------------------
# # Load ENV + GROQ
# # ------------------------------
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found in .env")
# groq_client = Groq(api_key=GROQ_API_KEY)

# # ------------------------------
# # Paths
# # ------------------------------
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# CHROMA_ROOT = PROJECT_ROOT / "app" / "db" / "chroma"
# USER_SESSIONS_ROOT = CHROMA_ROOT / "user_sessions"
# ONET_DB_PATH = CHROMA_ROOT / "csv_data"

# # ------------------------------
# # Config
# # ------------------------------
# class Config:
#     # MUST match ingest_user_data.py
#     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#     COLLECTION_PREFIX = "user_session"
#     TOP_K_SIMILAR = 10
#     FINAL_TOP_K = 5
#     GROQ_MODEL = "llama-3.1-8b-instant"


# # ------------------------------
# # Embeddings (cached)
# # ------------------------------
# _embeddings: Optional[HuggingFaceEmbeddings] = None


# def get_embeddings() -> HuggingFaceEmbeddings:
#     global _embeddings
#     if _embeddings is None:
#         device = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
#         _embeddings = HuggingFaceEmbeddings(
#             model_name=Config.EMBEDDING_MODEL,
#             model_kwargs={"device": device}
#         )
#         logger.info(f"Embeddings loaded: {Config.EMBEDDING_MODEL} [{device}]")
#     return _embeddings


# # ------------------------------
# # Find user Chroma directory
# # ------------------------------
# def find_user_chroma_dir(user_id: str) -> Optional[Path]:
#     user_dir = USER_SESSIONS_ROOT / user_id
#     if not user_dir.exists():
#         logger.warning(f"User directory not found: {user_dir}")
#         return None
#     sqlite_path = user_dir / "chroma.sqlite3"
#     if sqlite_path.exists():
#         logger.info(f"Found Chroma DB: {sqlite_path}")
#         return user_dir
#     logger.warning(f"chroma.sqlite3 not found in {user_dir}")
#     return None


# # ------------------------------
# # Load user-specific Chroma DB
# # ------------------------------
# def get_user_db(user_id: str) -> Optional[Chroma]:
#     chroma_dir = find_user_chroma_dir(user_id)
#     if not chroma_dir:
#         return None

#     persist_dir = str(chroma_dir)
#     embed_fn = get_embeddings()

#     try:
#         client = chromadb.PersistentClient(path=persist_dir)
#         collections = client.list_collections()
#     except Exception as e:
#         logger.error(f"Failed to connect to Chroma client: {e}")
#         return None

#     if not collections:
#         logger.warning("No collections found in user DB.")
#         return None

#     expected_name = f"{Config.COLLECTION_PREFIX}_{user_id}"
#     for coll in collections:
#         if coll.name != expected_name:
#             continue
#         try:
#             db = Chroma(
#                 collection_name=expected_name,
#                 embedding_function=embed_fn,
#                 persist_directory=persist_dir
#             )
#             count = db._collection.count()
#             if count > 0:
#                 logger.info(f"Loaded user collection '{expected_name}' ({count} docs)")
#                 return db
#         except Exception as e:
#             logger.error(f"Failed to load collection '{expected_name}': {e}")
#             return None

#     logger.warning(f"Collection '{expected_name}' not found or empty.")
#     return None


# # ------------------------------
# # Load ONET knowledge base
# # ------------------------------
# def get_onet_db() -> Optional[Chroma]:
#     if not ONET_DB_PATH.exists():
#         logger.error(f"ONET DB path not found: {ONET_DB_PATH}")
#         return None

#     try:
#         db = Chroma(
#             collection_name="career_knowledge_base",
#             embedding_function=get_embeddings(),
#             persist_directory=str(ONET_DB_PATH)
#         )
#         count = db._collection.count()
#         if count == 0:
#             logger.error("ONET DB is empty!")
#             return None
#         logger.info(f"ONET DB loaded: {count} careers")
#         return db
#     except Exception as e:
#         logger.error(f"Failed to load ONET DB: {e}")
#         return None


# # ------------------------------
# # Extract user insights
# # ------------------------------
# def extract_user_insights(db: Chroma) -> str:
#     try:
#         # Use a broad query to retrieve all user answers
#         docs = db.similarity_search("career interests skills education", k=500)
#         parts = []
#         for doc in docs:
#             content = doc.page_content.strip()
#             if content:
#                 parts.append(content)

#         if not parts:
#             return "No answers recorded."

#         logger.info(f"Extracted {len(parts)} user answer(s)")
#         return " | ".join(parts)

#     except Exception as e:
#         logger.error(f"Insight extraction failed: {e}")
#         return "Error reading user data."


# # ------------------------------
# # ONET similarity search
# # ------------------------------
# def match_with_onet(insights: str, onet_db: Chroma) -> List[Dict]:
#     try:
#         results = onet_db.similarity_search_with_score(insights, k=Config.TOP_K_SIMILAR)
#         matches = []
#         for doc, score in results:
#             title = doc.metadata.get("title", "Unknown Career")
#             desc = doc.page_content[:500]
#             sim = round(1 - score, 3)  # cosine distance → similarity
#             matches.append({"title": title, "desc": desc, "sim": sim})
#         logger.info(f"ONET raw matches: {len(matches)}")
#         return matches
#     except Exception as e:
#         logger.error(f"ONET similarity search failed: {e}")
#         return []


# # ------------------------------
# # Groq refinement (JSON output)
# # ------------------------------
# def match_with_groq(insights: str, onet_matches: List[Dict]) -> List[Dict]:
#     if not onet_matches:
#         return fallback_careers()

#     career_lines = [
#         f"- {m['title']}: {m['desc'][:300]}... (match: {m['sim']:.3f})"
#         for m in onet_matches
#     ]

#     prompt = f"""
# You are CareerCompass AI, helping 10th-grade Indian students choose careers.

# User's answers:
# {insights}

# Top ONET career matches:
# {chr(10).join(career_lines)}

# Return **exactly 3 to 5 careers** in **valid JSON** like this:

# [
#   {{
#     "title": "Software Engineer",
#     "match_score": 92,
#     "rationale": "You love coding and problem-solving...",
#     "roadmap": "11th-12th: PCM → JEE Main → B.Tech CSE",
#     "timeline": "2026-28: Prep | 2029-32: Degree | 2033+: Job"
#   }}
# ]

# Include Indian exams (JEE, NEET, CLAT, etc.) and realistic timelines.
# """

#     try:
#         resp = groq_client.chat.completions.create(
#             model=Config.GROQ_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=1200,
#             temperature=0.7,
#             response_format={"type": "json_object"}
#         )
#         raw = resp.choices[0].message.content.strip()
#         logger.debug(f"Groq raw response: {raw}")
#         return parse_groq_json(raw)
#     except Exception as e:
#         logger.error(f"Groq API error: {e}")
#         return fallback_careers(onet_matches)


# def parse_groq_json(text: str) -> List[Dict]:
#     import json
#     try:
#         data = json.loads(text)
#         if isinstance(data, list):
#             return data[:Config.FINAL_TOP_K]
#         elif isinstance(data, dict) and "careers" in data:
#             return data["careers"][:Config.FINAL_TOP_K]
#     except json.JSONDecodeError as e:
#         logger.warning(f"JSON parse failed: {e}. Falling back to line parser.")
#     return parse_groq_fallback(text)


# def parse_groq_fallback(text: str) -> List[Dict]:
#     """Fallback parser if JSON fails."""
#     matches = []
#     cur = {}
#     for line in text.split("\n"):
#         line = line.strip()
#         if line.startswith("Title:"):
#             if cur:
#                 matches.append(cur)
#             cur = {"title": line[6:].strip()}
#         elif line.startswith("Score:"):
#             try:
#                 cur["match_score"] = int(line.split("%")[0].split()[-1])
#             except:
#                 cur["match_score"] = 50
#         elif line.startswith("Rationale:"):
#             cur["rationale"] = line[10:].strip()
#         elif line.startswith("Roadmap:"):
#             cur["roadmap"] = line[8:].strip()
#         elif line.startswith("Timeline:"):
#             cur["timeline"] = line[9:].strip()
#     if cur:
#         matches.append(cur)
#     return matches[:Config.FINAL_TOP_K]


# # ------------------------------
# # Fallback careers
# # ------------------------------
# def fallback_careers(onet_matches: Optional[List[Dict]] = None) -> List[Dict]:
#     base = onet_matches[0] if onet_matches else {}
#     return [
#         {
#             "title": base.get("title", "Software Engineer"),
#             "match_score": max(40, int(base.get("sim", 0.5) * 100)),
#             "rationale": "Based on your interests and fallback analysis.",
#             "roadmap": "11th-12th: PCM → JEE Main → B.Tech in CS/IT",
#             "timeline": "2026–28: Prep | 2029–32: Degree | 2033+: Job"
#         }
#     ][:3]  # at least 1, max 3


# # ------------------------------
# # Main: match_careers
# # ------------------------------
# def match_careers(user_id: str) -> List[Dict]:
#     """
#     Main entry point used by Flask routes.
#     Returns list of 3–5 career dicts or error.
#     """
#     user_db = get_user_db(user_id)
#     if not user_db:
#         return [{"error": "No user data found. Please complete the assessment first."}]

#     onet_db = get_onet_db()
#     if not onet_db:
#         return [{"error": "Career database not available. Contact admin."}]

#     insights = extract_user_insights(user_db)
#     if "No answers" in insights or "Error" in insights:
#         return [{"error": "Not enough user responses to generate recommendations."}]

#     onet_matches = match_with_onet(insights, onet_db)
#     return match_with_groq(insights, onet_matches) or fallback_careers(onet_matches)


# # ------------------------------
# # Test function
# # ------------------------------
# def test():
#     test_user_id = "69084314081f22009861d0b4"  # Replace with real user if needed
#     print(f"\n{'='*60}")
#     print(f"CAREER MATCHES FOR USER: {test_user_id}")
#     print(f"{'='*60}\n")

#     results = match_careers(test_user_id)

#     if not results or "error" in results[0]:
#         print(f"ERROR: {results[0].get('error')}")
#         return

#     for i, m in enumerate(results, 1):
#         print(f"{i}. {m['title']}")
#         print(f"   Score: {m['match_score']}%")
#         print(f"   Why: {m['rationale']}")
#         print(f"   Plan: {m['roadmap']}")
#         print(f"   When: {m['timeline']}\n")


# if __name__ == "__main__":
#     test()



## Approach 2

"""
rag_utils.py – Weighted RAG + Groq for AI-CareerPath
Matches 10th-grade Indian students to 3–5 personalized career paths.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import warnings

# LangChain + Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

# Groq
from groq import Groq

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load ENV + GROQ
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")
groq_client = Groq(api_key=GROQ_API_KEY)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_ROOT = PROJECT_ROOT / "app" / "db" / "chroma"
USER_SESSIONS_ROOT = CHROMA_ROOT / "user_sessions"
ONET_DB_PATH = CHROMA_ROOT / "csv_data"

# Config
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_PREFIX = "user_session"
    TOP_K_SIMILAR = 12
    FINAL_TOP_K = 5
    GROQ_MODEL = "llama-3.1-8b-instant"

# Stage Weights (Priority: High to Low)
STAGE_WEIGHTS = {
    "aspirations": 1.0,      # Dream jobs, values → highest impact
    "interests": 0.9,        # Skills, hobbies → engagement
    "personality": 0.8,      # Work style, teamwork
    "academics": 0.7,        # Subjects, performance
    "preferences": 0.6,      # Location, time, risk
    "family_context": 0.4,   # Influence, not destiny
    "intro": 0.0             # Ignore
}

# Embeddings (cached)
_embeddings: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        device = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
        _embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": device}
        )
        logger.info(f"Embeddings loaded: {Config.EMBEDDING_MODEL} [{device}]")
    return _embeddings

# User DB: Per-user Chroma
def find_user_chroma_dir(user_id: str) -> Optional[Path]:
    user_dir = USER_SESSIONS_ROOT / user_id
    if not user_dir.exists():
        logger.warning(f"User directory not found: {user_dir}")
        return None
    sqlite_path = user_dir / "chroma.sqlite3"
    if sqlite_path.exists():
        logger.debug(f"Found Chroma DB: {sqlite_path}")
        return user_dir
    return None

def get_user_db(user_id: str) -> Optional[Chroma]:
    chroma_dir = find_user_chroma_dir(user_id)
    if not chroma_dir:
        return None

    persist_dir = str(chroma_dir)
    embed_fn = get_embeddings()
    expected_name = f"{Config.COLLECTION_PREFIX}_{user_id}"

    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
    except Exception as e:
        logger.error(f"Chroma client error: {e}")
        return None

    for coll in collections:
        if coll.name != expected_name:
            continue
        try:
            db = Chroma(
                collection_name=expected_name,
                embedding_function=embed_fn,
                persist_directory=persist_dir
            )
            count = db._collection.count()
            if count > 0:
                logger.info(f"Loaded user DB '{expected_name}' ({count} docs)")
                return db
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
    logger.warning(f"Collection '{expected_name}' not found or empty.")
    return None

# ONET DB: Shared Knowledge Base
def get_onet_db() -> Optional[Chroma]:
    if not ONET_DB_PATH.exists():
        logger.error(f"ONET DB path missing: {ONET_DB_PATH}")
        return None

    try:
        db = Chroma(
            collection_name="career_knowledge_base",
            embedding_function=get_embeddings(),
            persist_directory=str(ONET_DB_PATH)
        )
        count = db._collection.count()
        if count == 0:
            logger.error("ONET DB is empty!")
            return None
        logger.info(f"ONET DB loaded: {count} careers")
        return db
    except Exception as e:
        logger.error(f"ONET DB load failed: {e}")
        return None

# Extract + Weight User Insights
def extract_weighted_insights(db: Chroma) -> tuple[str, List[Dict]]:
    try:
        docs = db.similarity_search("career skills interests values education", k=500)
        weighted_parts = []
        evidence = []

        for doc in docs:
            content = doc.page_content.strip()
            if not content:
                continue
            meta = doc.metadata
            stage = meta.get("stage", "unknown")
            weight = STAGE_WEIGHTS.get(stage, 0.5)

            # Repeat high-weight answers to boost signal
            repeat = 3 if weight >= 0.8 else 2 if weight >= 0.6 else 1
            weighted_parts.extend([content] * repeat)

            evidence.append({
                "text": content[:150],
                "stage": stage,
                "weight": round(weight, 2),
                "source": f"{stage.capitalize()} response"
            })

        insight_text = " | ".join(weighted_parts[:50])  # Limit context
        logger.info(f"Extracted {len(weighted_parts)} weighted insights")
        return insight_text, evidence[:10]

    except Exception as e:
        logger.error(f"Insight extraction failed: {e}")
        return "No user data.", []


# ONET Similarity Search
def match_with_onet(insights: str, onet_db: Chroma) -> List[Dict]:
    try:
        results = onet_db.similarity_search_with_score(insights, k=Config.TOP_K_SIMILAR)
        matches = []
        for doc, score in results:
            title = doc.metadata.get("title", "Unknown Career")
            desc = doc.page_content[:600]
            sim = round(1 - score, 3)
            matches.append({"title": title, "desc": desc, "sim": sim})
        logger.info(f"ONET retrieved {len(matches)} candidates")
        return matches
    except Exception as e:
        logger.error(f"ONET search failed: {e}")
        return []

# Groq Refinement (JSON + Explainability)
def refine_with_groq(insights: str, onet_matches: List[Dict], evidence: List[Dict]) -> List[Dict]:
    if not onet_matches:
        return fallback_careers()

    career_lines = [
        f"- {m['title']}: {m['desc'][:250]}... (sim: {m['sim']:.3f})"
        for m in onet_matches
    ]

    evidence_snippets = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence[:5]])

    prompt = f"""
You are **CareerCompass AI**, guiding 10th-grade Indian students.

**User Profile (weighted):**
{insights}

**Top ONET Matches:**
{chr(10).join(career_lines)}

**Key Evidence from User:**
{evidence_snippets}

---

Return **exactly 3–5 careers** in **valid JSON**:

[
  {{
    "title": "Software Engineer",
    "match_score": 94,
    "rationale": "You love coding and want high income...",
    "roadmap": "11th-12th: PCM → JEE Main → B.Tech CSE",
    "timeline": "2026–28: Prep | 2029–32: Degree | 2033+: Job"
  }}
]

- Use **Indian exams**: JEE, NEET, CLAT, NATA, etc.
- Timeline: 2026 onward
- Rationale: Reference user aspirations, skills, values
- Score: 0–100 (higher = better fit)
"""

    try:
        resp = groq_client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content.strip()
        logger.debug(f"Groq response: {raw[:500]}...")
        return parse_groq_json(raw)
    except Exception as e:
        logger.error(f"Groq failed: {e}")
        return fallback_careers(onet_matches)

def parse_groq_json(text: str) -> List[Dict]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data[:Config.FINAL_TOP_K]
        elif isinstance(data, dict) and "careers" in data:
            return data["careers"][:Config.FINAL_TOP_K]
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
    return parse_fallback(text)

def parse_fallback(text: str) -> List[Dict]:
    matches = []
    cur = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("Title:", '"title"')):
            if cur: matches.append(cur)
            cur = {"title": line.split(":", 1)[1].strip().strip('"')}
        elif "match_score" in line or line.startswith("Score:"):
            try: cur["match_score"] = int(''.join(filter(str.isdigit, line)))
            except: cur["match_score"] = 60
        elif line.startswith(("Rationale:", '"rationale"')):
            cur["rationale"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith(("Roadmap:", '"roadmap"')):
            cur["roadmap"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith(("Timeline:", '"timeline"')):
            cur["timeline"] = line.split(":", 1)[1].strip().strip('"')
    if cur: matches.append(cur)
    return matches[:Config.FINAL_TOP_K]

# Fallback Careers
def fallback_careers(onet_matches: Optional[List[Dict]] = None) -> List[Dict]:
    base = (onet_matches or [{}])[0]
    title = base.get("title", "Software Engineer")
    sim = base.get("sim", 0.5)
    return [
        {
            "title": title,
            "match_score": max(40, int(sim * 100)),
            "rationale": "Recommended based on your interests and available data.",
            "roadmap": "11th-12th: PCM → JEE Main → B.Tech",
            "timeline": "2026–28: Prep | 2029–32: Degree | 2033+: Job"
        }
    ]


# Main: match_careers
def match_careers(user_id: str) -> List[Dict]:
    """
    Entry point for Flask.
    Returns 3–5 career dicts or error.
    """
    user_db = get_user_db(user_id)
    if not user_db:
        return [{"error": "No user data. Please complete the assessment."}]

    onet_db = get_onet_db()
    if not onet_db:
        return [{"error": "Career database unavailable. Contact support."}]

    insights, evidence = extract_weighted_insights(user_db)
    if "No answers" in insights:
        return [{"error": "Not enough responses to recommend careers."}]

    onet_matches = match_with_onet(insights, onet_db)
    final_careers = refine_with_groq(insights, onet_matches, evidence)

    if not final_careers or len(final_careers) == 0:
        final_careers = fallback_careers(onet_matches)

    logger.info(f"Final {len(final_careers)} career(s) for user {user_id}")
    return final_careers

# Test
def test():
    test_id = "6907852acce533e5a71cc9e2"  # Replace with real user
    print(f"\n{'='*70}")
    print(f"CAREER RECOMMENDATIONS FOR USER: {test_id}")
    print(f"{'='*70}\n")
    results = match_careers(test_id)
    if "error" in results[0]:
        print(f"ERROR: {results[0]['error']}")
    else:
        for i, c in enumerate(results, 1):
            print(f"{i}. {c['title']} ({c['match_score']}%)")
            print(f"   Why: {c['rationale']}")
            print(f"   Plan: {c['roadmap']}")
            print(f"   When: {c['timeline']}\n")

if __name__ == "__main__":
    test()