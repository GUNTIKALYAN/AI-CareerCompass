from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from bson import ObjectId
from pymongo import MongoClient
from app.utils.db_utils import users_collection,create_user, verify_user, update_last_login,create_career_chat_history


auth_bp = Blueprint("auth_bp", __name__, url_prefix="/auth")

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["Ai_career_roadmap"]

users_collection = db["users"]
chat_histories_collection = db["chat_histories"]


# --- Helper ---
def create_career_chat_history(user_id, grade="10"):
    chat_history = {
        "user_id": ObjectId(user_id),
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
                "interests": False,
                "aspirations": False,
                "preferences": False,
                "complete": False,
            },
        },
    }
    chat_histories_collection.insert_one(chat_history)


# --- Routes ---

# Home (login/signup page)
@auth_bp.route("/test")
def test():
    return "Auth blueprint works!"


@auth_bp.route("/")
def home():
    return render_template("auth/login.html")


# Signup
@auth_bp.route("/signup", methods=["POST"])
def signup():
    username = request.form["username"].strip()
    email = request.form["email"].strip().lower()
    password = request.form["password"]
    confirm_password = request.form["confirm_password"]

    if password != confirm_password:
        flash("Passwords do not match!", "error")
        return redirect(url_for("auth_bp.home"))

    if users_collection.find_one({"email": email}):
        flash("Email already registered. Please login.", "error")
        return redirect(url_for("auth_bp.home"))

    if len(password) < 6:
        flash("Password must be at least 6 characters long!", "error")
        return redirect(url_for("auth_bp.home"))

    hashed_pw = generate_password_hash(password)
    user_doc = {
        "username": username,
        "email": email,
        "password": hashed_pw,
        "initial_assessment_done": False,
        "created_at": datetime.datetime.now(),
        "last_login": None,
    }
    result = users_collection.insert_one(user_doc)

    # initialize chat session
    create_career_chat_history(result.inserted_id, grade="10")

    flash("Signup successful! Please log in.", "success")
    return redirect(url_for("auth_bp.home"))


# Login
@auth_bp.route("/login", methods=["POST"])
def login():
    email = request.form["email"].strip().lower()
    password = request.form["password"]

    user = users_collection.find_one({"email": email})
    if not user:
        flash("Email not registered. Signup first.", "error")
        return redirect(url_for("auth_bp.home"))

    if not check_password_hash(user["password"], password):
        flash("Incorrect password. Try again.", "error")
        return redirect(url_for("auth_bp.home"))

    session["user"] = user["username"]
    session["user_id"] = str(user["_id"])

    update_last_login(user["_id"])

    # If not finished assessment → go to chat
    if not user.get("initial_assessment_done", False):
        flash(f"Welcome {user['username']}! Let's start with a quick career assessment.", "info")
        return redirect(url_for("career_bp.index"))

    # Otherwise → go to dashboard
    flash(f"Welcome back, {user['username']}!", "success")
    return redirect(url_for("dashboard_bp.index"))


# Logout
@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("auth_bp.home"))
