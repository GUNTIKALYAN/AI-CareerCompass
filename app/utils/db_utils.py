from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
import datetime
from bson.objectid import ObjectId
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "Ai_career_roadmap"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Define collections
users_collection = db.users
chat_histories_collection = db.chat_histories
career_profiles_collection = db.career_profiles
career_matches_collection = db.career_matches  # Added missing collection

def create_user(name, email, password):
    if users_collection.find_one({"email": email}):
        return False, "Email already exists"
    user = {
        "username": name,
        "email": email,
        "password": generate_password_hash(password),
        "initial_assessment_done": False,
        "created_at": datetime.datetime.now(),
        "last_login": None
    }
    result = users_collection.insert_one(user)
    user_id = str(result.inserted_id)
    create_career_chat_history(user_id)
    create_career_profile(user_id, name, email)
    logger.info(f"Created user: {user_id}, Email: {email}")
    return True, user_id

def verify_user(email, password):
    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        logger.warning(f"Failed to verify user: Email: {email}")
        return None
    user_id = str(user["_id"])
    logger.info(f"Verified user: {user_id}")
    return user_id

def update_last_login(user_id):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    result = users_collection.update_one(
        {"_id": obj_id},
        {"$set": {"last_login": datetime.datetime.now()}}
    )
    logger.info(f"Updated last login: {user_id}, Modified: {result.modified_count}")
    return result.modified_count > 0

def create_career_profile(user_id, name, email):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    profile = {
        "user_id": obj_id,
        "username": name,
        "email": email,
        "created_at": datetime.datetime.now(),
        "updated_at": datetime.datetime.now(),
        "initial_assessment_done": False,
        "academic_info": {},
        "family_background": {},
        "interests_skills": {},
        "career_aspirations": {},
        "practical_factors": {},
        "personality_traits": {},
        "ai_insights": {}
    }
    result = career_profiles_collection.insert_one(profile)
    logger.info(f"Created career profile: {user_id}, ID: {result.inserted_id}")
    return result.acknowledged

def get_career_profile(user_id):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return None
    return career_profiles_collection.find_one({"user_id": obj_id})

def update_career_profile_section(user_id, section, data):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    result = career_profiles_collection.update_one(
        {"user_id": obj_id},
        {"$set": {f"{section}": data, "updated_at": datetime.datetime.now()}}
    )
    logger.info(f"Updated profile section: {user_id}, Section: {section}, Modified: {result.modified_count}")
    return result.modified_count > 0

def analyze_and_update_insights(user_id, chat_messages):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    insights = extract_insights_from_chat(chat_messages)
    result = career_profiles_collection.update_one(
        {"user_id": obj_id},
        {"$set": {**insights, "initial_assessment_done": True, "updated_at": datetime.datetime.now()}}
    )
    logger.info(f"Updated insights: {user_id}, Modified: {result.modified_count}")
    return result.modified_count > 0

def extract_insights_from_chat(messages):
    insights = {
        "academic_info": {},
        "family_background": {},
        "interests_skills": {},
        "career_aspirations": {},
        "practical_factors": {},
        "personality_traits": {}
    }
    for message in messages:
        user_msg = message.get("user_message", "").lower()
        stage = message.get("stage", "")
        if stage == "academics":
            insights["academic_info"].setdefault("responses", []).append(user_msg)
        elif stage == "family_context":
            insights["family_background"].setdefault("responses", []).append(user_msg)
        elif stage == "interests":
            insights["interests_skills"].setdefault("responses", []).append(user_msg)
        elif stage == "aspirations":
            insights["career_aspirations"].setdefault("responses", []).append(user_msg)
        elif stage == "preferences":  # Corrected from "practical"
            insights["practical_factors"].setdefault("responses", []).append(user_msg)
        elif stage == "personality":
            insights["personality_traits"].setdefault("responses", []).append(user_msg)
    return insights

def create_career_chat_history(user_id, grade="10"):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    if chat_histories_collection.find_one({"user_id": obj_id}):
        logger.info(f"Chat history already exists for user_id: {user_id}")
        return True
    chat_history = {
        "user_id": obj_id,
        "created_at": datetime.datetime.now(),
        "updated_at": datetime.datetime.now(),
        "career_session": {
            "grade": grade,
            "current_stage": "intro",
            "completed": False,
            "messages": [],
            "stage_completion": {
                "intro": False,
                "academics": False,
                "family_context": False,
                "interests": False,
                "aspirations": False,
                "preferences": False,
                "complete": False
            }
        }
    }
    result = chat_histories_collection.insert_one(chat_history)
    logger.info(f"Created chat history: {user_id}, ID: {result.inserted_id}")
    return result.acknowledged

def get_career_chat(user_id):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return None
    chat_doc = chat_histories_collection.find_one({"user_id": obj_id})
    if not chat_doc:
        logger.info(f"No chat history for user_id: {user_id}, creating...")
        create_career_chat_history(user_id)
        chat_doc = chat_histories_collection.find_one({"user_id": obj_id})
    if chat_doc and "career_session" in chat_doc:
        logger.info(f"Retrieved chat history: {user_id}, Messages: {len(chat_doc['career_session']['messages'])}")
        return chat_doc["career_session"]
    logger.warning(f"Fallback chat history for user_id: {user_id}")
    return {
        "grade": "10",
        "current_stage": "intro",
        "completed": False,
        "messages": [],
        "stage_completion": {
            "intro": False,
            "academics": False,
            "family_context": False,
            "interests": False,
            "aspirations": False,
            "preferences": False,
            "complete": False
        }
    }

def get_career_stage(user_id):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return "intro"
    chat = get_career_chat(user_id)
    stage = chat.get("current_stage", "intro") if chat else "intro"
    logger.info(f"Retrieved stage: {user_id}, Stage: {stage}")
    return stage

def update_career_stage(user_id, stage):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    current_stage = get_career_stage(user_id)
    stages = ["intro", "academics", "family_context", "interests", "aspirations", "preferences", "complete"]
    progress = (stages.index(stage) / (len(stages) - 1)) * 100 if stage in stages else 0
    result = chat_histories_collection.update_one(
        {"user_id": obj_id},
        {
            "$set": {
                "career_session.current_stage": stage,
                f"career_session.stage_completion.{current_stage}": True,
                "career_session.progress_percentage": progress,
                "updated_at": datetime.datetime.now()
            }
        },
        upsert=True
    )
    logger.info(f"Update stage: {user_id}, Stage: {stage}, Matched: {result.matched_count}, Modified: {result.modified_count}")
    return result.modified_count > 0 or result.upserted_id

def save_career_interaction(user_id, user_message, ai_response, stage=None):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    if stage is None:
        stage = get_career_stage(user_id)
    message_pair = {
        "timestamp": datetime.datetime.now(),
        "user_message": user_message,
        "ai_response": ai_response,
        "stage": stage,
        "message_length": len(user_message),
        "response_quality": None,
        "extracted_keywords": []
    }
    result = chat_histories_collection.update_one(
        {"user_id": obj_id},
        {
            "$push": {"career_session.messages": message_pair},
            "$set": {"updated_at": datetime.datetime.now()}
        },
        upsert=True
    )
    logger.info(f"Save interaction: {user_id}, Message: '{user_message}', Stage: {stage}, Matched: {result.matched_count}, Modified: {result.modified_count}")
    return result.modified_count > 0 or result.upserted_id

def mark_career_complete(user_id):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return False
    chat_doc = chat_histories_collection.find_one({"user_id": obj_id})
    if chat_doc and "career_session" in chat_doc:
        messages = chat_doc["career_session"].get("messages", [])
        analyze_and_update_insights(user_id, messages)
    result = chat_histories_collection.update_one(
        {"user_id": obj_id},
        {
            "$set": {
                "career_session.completed": True,
                "career_session.current_stage": "complete",
                "career_session.progress_percentage": 100,
                "career_session.stage_completion.complete": True,
                "updated_at": datetime.datetime.now()
            }
        },
        upsert=True
    )
    users_collection.update_one(
        {"_id": obj_id},
        {"$set": {"initial_assessment_done": True}}
    )
    logger.info(f"Mark complete: {user_id}, Matched: {result.matched_count}, Modified: {result.modified_count}")
    return result.modified_count > 0 or result.upserted_id

def get_user_progress_stats(user_id):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return None
    chat_doc = chat_histories_collection.find_one({"user_id": obj_id})
    profile_doc = career_profiles_collection.find_one({"user_id": obj_id})
    if not chat_doc:
        logger.warning(f"No chat history: {user_id}")
        return None
    career_session = chat_doc.get("career_session", {})
    stats = {
        "progress_percentage": career_session.get("progress_percentage", 0),
        "current_stage": career_session.get("current_stage", "intro"),
        "completed_stages": sum(1 for completed in career_session.get("stage_completion", {}).values() if completed),
        "total_messages": len(career_session.get("messages", [])),
        "assessment_completed": career_session.get("completed", False),
        "profile_data_available": profile_doc is not None and profile_doc.get("initial_assessment_done", False),
        "last_activity": chat_doc.get("updated_at")
    }
    logger.info(f"Progress stats: {user_id}, Stats: {stats}")
    return stats

def get_user_answers(user_id):
    try:
        obj_id = ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return []
    chat_doc = chat_histories_collection.find_one({"user_id": obj_id})
    if not chat_doc or "career_session" not in chat_doc:
        logger.warning(f"No chat history or session for user_id: {user_id}")
        return []
    messages = chat_doc["career_session"].get("messages", [])
    user_answers = [msg["user_message"] for msg in messages if "user_message" in msg]
    logger.info(f"Retrieved {len(user_answers)} answers for user_id: {user_id}")
    return user_answers

def save_career_matches(user_id: str, career_matches: list) -> bool:
    """Save career matches to the database."""
    try:
        obj_id = ObjectId(user_id)
        career_matches_collection.update_one(
            {"user_id": user_id},
            {"$set": {"matches": career_matches, "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )
        logger.info(f"Saved career matches for user_id: {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save career matches for user_id {user_id}: {e}")
        return False

def get_career_matches(user_id: str) -> list:
    """Retrieve career matches from the database."""
    try:
        match_doc = career_matches_collection.find_one({"user_id": user_id})
        matches = match_doc.get("matches", []) if match_doc else []
        logger.info(f"Retrieved career matches for user_id: {user_id}, Matches count: {len(matches)}")
        return matches
    except Exception as e:
        logger.error(f"Failed to retrieve career matches for user_id {user_id}: {e}")
        return []