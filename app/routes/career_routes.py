from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for
from app.scripts.ingest_user_data import ingest_user_answers
from app.utils.rag_utils import match_careers
from app.utils.db_utils import (
    get_career_chat,
    get_career_stage,
    update_career_stage,
    save_career_interaction,
    mark_career_complete,
)
from bson.objectid import ObjectId
import os
import logging
from groq import Groq
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

career_bp = Blueprint("career_bp", __name__, url_prefix="/career")

# Initialize Groq client
API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")
llama_client = Groq(api_key=API_KEY)

# System prompt for AI questioning
SYSTEM_PROMPT = """
You are CareerCompass AI, an expert career counselor specializing in creating personalized career roadmaps. Your goal is to gather comprehensive information through strategic questioning to build a smart career path for the user.

QUESTIONING STRATEGY:
Follow these stages in order, asking 1-2 questions per stage:

1. ACADEMICS (Stage: academics)
   - Subjects enjoyment and performance
   - Learning style (theoretical vs practical)

2. CAREER INTEREST (Stage: career_interest)
   - Long-term goals and dream jobs
   - Fields/industries of interest
   - Role models and inspirations

3. SKILLS AND STRENGTHS (Stage: skills_strengths)
   - Hobbies and extracurricular passions
   - Confident personal skills
   - Skills to develop

4. LEARNING STYLE AND PREFERENCES (Stage: learning_preferences)
   - Solo vs group learning
   - Best ways to absorb information

5. PRACTICAL CONSIDERATIONS AND CONSTRAINTS (Stage: practical_constraints)
   - Financial, geographical, family constraints
   - Openness to relocation

6. MOTIVATION AND VALUES (Stage: motivation_values)
   - Key motivators
   - Core job values

7. FAMILY CONTEXT (Stage: family)
   - Family attitude toward education/career
   - Financial limitations/responsibilities
   - Emotional/financial support
   - Family support for relocation

8. EVALUATION AND FEEDBACK (Stage: evaluation)
   - Existing career plans/advice received
   - Preference for dynamic roadmap

RESPONSE GUIDELINES:
- Ask ONE clear, specific question at a time
- Be conversational and empathetic
- Adapt follow-up questions based on previous answers
- Show genuine interest in their responses
- Vary your questions within each stage to explore different aspects - don't repeat similar questions
- After 10-12 meaningful exchanges, conclude the assessment

COMPLETION MESSAGE:
"Thank you {user_name}! I've gathered valuable insights about your background, interests, and aspirations. Based on our conversation, I'll now prepare personalized career recommendations for you. You'll be redirected to your dashboard where you can explore your customized career roadmap!"

Stay focused on information gathering - do not provide career advice during the questioning phase.
"""

STAGE_QUESTIONS = {
    "academics": [
        "What subjects do you enjoy studying the most and why?",
        "What are your strongest subjects or grades currently?",
        "Do you prefer theoretical subjects or practical ones?"
    ],
    "career_interest": [
        "What are your long-term career goals or dream jobs, if any?",
        "Are there any specific fields or industries that interest you (e.g., engineering, medicine, arts, technology, business, etc.)?",
        "Do you admire any professionals or career role models? What about their journeys attracts you?"
    ],
    "skills_strengths": [
        "What extra activities or hobbies are you passionate about?",
        "Which personal skills do you feel most confident in (e.g., problem-solving, creativity, teamwork, leadership)?",
        "Are there skills you'd like to learn or improve for your future career?"
    ],
    "learning_preferences": [
        "Do you prefer learning alone or in groups?",
        "How do you best absorb new information (reading, videos, interactive activities)?"
    ],
    "practical_constraints": [
        "Are there any constraints (financial, geographical, family) that may impact your career choices?",
        "Are you open to relocating for education or career opportunities?"
    ],
    "motivation_values": [
        "What motivates you to achieve your goals (recognition, financial stability, impact, personal growth)?",
        "What values are most important to you in a future job (flexibility, social impact, innovation)?"
    ],
    "family": [
        "What is your family's attitude toward your higher education or chosen career path?",
        "Are there any financial limitations or responsibilities that may affect your career choices?",
        "Do you have family members who can support you emotionally or financially in your career journey?",
        "Is relocation for studies or work an option your family would support?"
    ],
    "evaluation": [
        "Do you already have any career plans or have you received career advice before? How did it help or fail to help?",
        "Would you prefer a career roadmap that adapts dynamically based on your progress and changing preferences?"
    ]
}

@career_bp.route("/")
def index():
    """Render the chat interface with an initial admin welcome message."""
    if "user_id" not in session:
        logger.warning("No user_id in session, redirecting to login")
        return redirect(url_for("auth_bp.home"))
    
    user_id = session["user_id"]
    user_name = session.get("user", "User")
    
    try:
        ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return jsonify({"error": "Invalid user ID"}), 400
    
    chat = get_career_chat(user_id)
    current_stage = get_career_stage(user_id)
    chat_history = chat.get("messages", [])
    
    # Initialize chat with admin welcome message if empty
    if not chat_history and current_stage == "intro":
        welcome_message = f"Hi {user_name}, I'm CareerCompass AI! I'm here to help you explore your career path. Can you please answer some questions to start building your personalized career roadmap?"
        success = save_career_interaction(user_id, "", welcome_message, "intro")
        if not success:
            logger.warning(f"Failed to save welcome message for user_id: {user_id}")
        chat_history = [{"user_message": "", "ai_response": welcome_message, "stage": "intro"}]
    
    return render_template(
        "career/chat.html",
        user_name=user_name,
        user_id=user_id,
        chat_history=chat_history,
        stage=current_stage,
        assessment_complete=chat.get("completed", False)
    )

def generate_ai_question(user_name: str, user_id: str, user_message: str, chat_history: List[Dict], current_stage: str, message_count: int) -> str:
    """Generate the next AI question based on stage and user response."""
    logger.debug(f"Generating question: user_id={user_id}, stage={current_stage}, message_count={message_count}, user_message='{user_message}'")
    
    # Handle first user response after welcome message
    if current_stage == "intro" and user_message:
        success = update_career_stage(user_id, "academics")
        if not success:
            logger.warning(f"Failed to update stage to academics for user_id: {user_id}")
        return "Great, let's dive in! What grade are you currently in, and which subjects do you find most interesting or perform best in?"
    
    # Get stage-specific question index (cycle through questions for variety)
    questions_per_stage = len(STAGE_QUESTIONS.get(current_stage, []))
    if questions_per_stage > 0 and not user_message:
        question_index = (message_count - 1) % questions_per_stage
        return STAGE_QUESTIONS[current_stage][question_index]
    
    # Generate adaptive follow-up based on user response
    chat_history_text = "\n".join(
        [f"User: {m['user_message']}\nAI: {m['ai_response']}" for m in chat_history]
    )
    prompt = f"""
{SYSTEM_PROMPT}

Current stage: {current_stage}
Previous conversation:
{chat_history_text}

User's latest response: {user_message}

Based on their response, ask ONE specific follow-up question that:
1. Acknowledges what they shared in a personalized way
2. Digs deeper into a different aspect of the {current_stage} topic (vary from previous questions in this stage)
3. Is conversational, empathetic, and engaging
4. Helps gather new information for career guidance

Keep it to 1-2 sentences max. Do not repeat previous questions in this stage. Be creative in phrasing to keep the conversation fresh.
"""
    try:
        response = llama_client.chat.completions.create(
            model="llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.8
        )
        ai_text = response.choices[0].message.content.strip()
        logger.debug(f"Generated AI response: {ai_text}")
        return ai_text
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        # Fallback to template question
        fallback_index = (message_count - 1) % questions_per_stage if questions_per_stage > 0 else 0
        return STAGE_QUESTIONS.get(current_stage, ["Can you tell me more about your interests?"])[fallback_index]

def determine_next_stage(current_stage: str, message_count: int) -> str:
    """Determine the next stage based on message count."""
    stage_transitions = {
        "intro": ("academics", 2),
        "academics": ("family_context", 4),
        "family_context": ("interests", 6),
        "interests": ("aspirations", 8),
        "aspirations": ("preferences", 10),
        "preferences": ("personality", 12),
        "personality": ("complete", 14)
    }
    if current_stage in stage_transitions:
        next_stage, threshold = stage_transitions[current_stage]
        if message_count >= threshold:
            logger.debug(f"Stage transition: {current_stage} -> {next_stage} at message {message_count}")
            return next_stage
    return current_stage

@career_bp.route("/chat", methods=["POST"])
def chat():
    """Handle user chat messages and generate AI responses."""
    if "user_id" not in session:
        logger.warning("No user_id in session")
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = session["user_id"]
    user_name = session.get("user", "User")
    
    try:
        ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return jsonify({"error": "Invalid user ID"}), 400
    
    data = request.json
    user_message = data.get("message", "").strip()
    
    chat_history_doc = get_career_chat(user_id)
    current_stage = get_career_stage(user_id)
    chat_history = chat_history_doc.get("messages", [])
    message_count = len(chat_history) + (1 if user_message else 0)
    
    logger.debug(f"Chat: user_id={user_id}, stage={current_stage}, message_count={message_count}, user_message='{user_message}'")
    
    # Handle user response
    if user_message:
        # Ingest user answer into Chroma
        try:
            success, chunk_count = ingest_user_answers(user_id, answers=user_message, answer_id=f"answer_{message_count}")
            if not success:
                logger.warning(f"Failed to ingest answer for user_id: {user_id}, message: {user_message}")
        except Exception as e:
            logger.error(f"Error ingesting answer for user_id {user_id}: {e}")
        
        # Determine next stage
        next_stage = determine_next_stage(current_stage, message_count)
        if next_stage != current_stage:
            success = update_career_stage(user_id, next_stage)
            if success:
                current_stage = next_stage
            else:
                logger.warning(f"Failed to update stage to {next_stage} for user_id: {user_id}")
        
        # Generate AI response
        ai_response = generate_ai_question(user_name, user_id, user_message, chat_history, current_stage, message_count)
        success = save_career_interaction(user_id, user_message, ai_response, current_stage)
        if not success:
            logger.warning(f"Failed to save interaction for user_id: {user_id}")
        
        # Check for assessment completion
        assessment_complete = message_count >= 14 or current_stage == "complete"
        if assessment_complete:
            mark_career_complete(user_id)
            completion_message = f"Thank you {user_name}! I've gathered valuable insights about your background, interests, and aspirations. Based on our conversation, I'll now prepare personalized career recommendations for you. You'll be redirected to your dashboard where you can explore your customized career roadmap!"
            return jsonify({
                "ai_response": completion_message,
                "stage": "complete",
                "assessment_complete": True,
                "redirect_url": url_for("dashboard_bp.index")
            })
        
        return jsonify({
            "ai_response": ai_response,
            "stage": current_stage,
            "assessment_complete": False
        })
    
    return jsonify({"error": "No message provided"}), 400

@career_bp.route("/complete", methods=["POST"])
def complete_assessment():
    """Manually complete the career assessment."""
    if "user_id" not in session:
        logger.warning("No user_id in session")
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = session["user_id"]
    try:
        ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return jsonify({"error": "Invalid user ID"}), 400
    
    mark_career_complete(user_id)
    return jsonify({
        "message": "Assessment completed",
        "redirect_url": url_for("dashboard_bp.index")
    })

@career_bp.route("/insights", methods=["GET"])
def get_career_insights():
    """Retrieve career insights for the user."""
    if "user_id" not in session:
        logger.warning("No user_id in session")
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = session["user_id"]
    try:
        ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return jsonify({"error": "Invalid user ID"}), 400
    
    chat = get_career_chat(user_id)
    if not chat or not chat.get("completed", False):
        return jsonify({"error": "Assessment not completed"}), 400
    
    messages = chat.get("messages", [])
    insights = analyze_career_data(messages)

    career_matches = session.get('career_matches')
    if not career_matches:
        try:
            career_matches = match_careers(user_id)
            if not any("error" in m for m in career_matches):
                session['career_matches'] = career_matches
            else:
                logger.warning(f"Career matches computation failed for user_id: {user_id}")
                career_matches = []
        except Exception as e:
            logger.error(f"Error recomputing career matches for user_id {user_id}: {e}")
            career_matches = []

    return jsonify({
        "insights": insights,
        "career_matches": career_matches,
        "total_interactions": len(messages),
        "completion_date": chat.get("updated_at")
    })

@career_bp.route("/career_matching", methods=["GET"])
def get_career_matching():
    """Retrieve or compute career matches for the user."""
    if "user_id" not in session:
        logger.warning("No user_id in session")
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = session["user_id"]
    try:
        ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return jsonify({"error": "Invalid user ID"}), 400
    
    chat = get_career_chat(user_id)
    if not chat or not chat.get("completed", False):
        return jsonify({"error": "Assessment not completed"}), 400
    
    career_matches = session.get('career_matches')
    if not career_matches:
        try:
            career_matches = match_careers(user_id)
            if not any("error" in m for m in career_matches):
                session['career_matches'] = career_matches
            else:
                logger.warning(f"Career matches computation failed for user_id: {user_id}")
                career_matches = []
        except Exception as e:
            logger.error(f"Error computing career matches for user_id {user_id}: {e}")
            career_matches = []
    
    return jsonify({
        "career_matches": career_matches,
        "message": "Career matches retrieved successfully"
    })

@career_bp.route("/career_path", methods=["GET"])
def get_career_path():
    """Generate a personalized career roadmap for the user."""
    if "user_id" not in session:
        logger.warning("No user_id in session")
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = session["user_id"]
    try:
        ObjectId(user_id)
    except Exception as e:
        logger.error(f"Invalid user_id: {user_id}, Exception: {e}")
        return jsonify({"error": "Invalid user ID"}), 400
    
    chat = get_career_chat(user_id)
    if not chat or not chat.get("completed", False):
        return jsonify({"error": "Assessment not completed"}), 400
    
    messages = chat.get("messages", [])
    insights = analyze_career_data(messages)
    
    career_matches = session.get('career_matches')
    if not career_matches:
        try:
            career_matches = match_careers(user_id)
            if not any("error" in m for m in career_matches):
                session['career_matches'] = career_matches
            else:
                logger.warning(f"Career matches computation failed for user_id: {user_id}")
                career_matches = []
        except Exception as e:
            logger.error(f"Error computing career matches for user_id {user_id}: {e}")
            career_matches = []
    
    # Generate career roadmap (simplified example)
    roadmap = {
        "steps": [
            {"step": "Education", "description": f"Based on your academic profile ({insights['academic_profile'].get('responses', [''])}), consider pursuing relevant courses or degrees in {', '.join(career_matches[:2]) if career_matches else 'your top interests'}."},
            {"step": "Skill Development", "description": f"Develop skills in {', '.join(insights['interests_skills'].get('responses', ['']))} to align with {career_matches[0] if career_matches else 'your career goals'}."},
            {"step": "Networking", "description": f"Connect with professionals in {career_matches[0] if career_matches else 'your chosen field'} through online platforms or local events."},
            {"step": "Application", "description": f"Apply for internships or entry-level roles in {career_matches[0] if career_matches else 'your target industry'} within the next 6-12 months."}
        ],
        "resources": [
            "Online courses on Coursera or edX",
            "Career fairs or workshops",
            "LinkedIn networking groups"
        ],
        "timeline": "6-18 months"
    }
    
    return jsonify({
        "roadmap": roadmap,
        "insights_summary": insights,
        "career_matches": career_matches,
        "completion_date": chat.get("updated_at")
    })


def analyze_career_data(messages: List[Dict]) -> Dict:
    """Analyze chat messages to generate career insights."""
    insights = {
        "academic_profile": {},
        "family_background": {},
        "interests_skills": {},
        "career_preferences": {},
        "personality_traits": {},
        "practical_considerations": {}
    }
    for message in messages:
        user_msg = message.get("user_message", "").lower()
        stage = message.get("stage", "")
        if stage == "academics":
            insights["academic_profile"].setdefault("responses", []).append(user_msg)
        elif stage == "family_context":
            insights["family_background"].setdefault("responses", []).append(user_msg)
        elif stage == "interests":
            insights["interests_skills"].setdefault("responses", []).append(user_msg)
        elif stage == "aspirations":
            insights["career_preferences"].setdefault("responses", []).append(user_msg)
        elif stage == "preferences":
            insights["practical_considerations"].setdefault("responses", []).append(user_msg)
        elif stage == "personality":
            insights["personality_traits"].setdefault("responses", []).append(user_msg)
    return insights


