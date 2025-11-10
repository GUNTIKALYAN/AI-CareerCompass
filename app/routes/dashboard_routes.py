# app/routes/dashboard_routes.py
from flask import Blueprint, render_template, session, redirect, url_for, request, jsonify
from app.utils.rag_utils import match_careers

dashboard_bp = Blueprint("dashboard_bp", __name__, url_prefix="/dashboard")

@dashboard_bp.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("auth_bp.home"))
    user_name = session.get("user", "Student")
    return render_template("dashboard/dashboard.html", user_name=user_name)


@dashboard_bp.route('/ai_matching')
def get_ai_matching():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    careers = match_careers(user_id)

    # Handle RAG errors
    if isinstance(careers, list) and careers and "error" in careers[0]:
        return jsonify({"error": careers[0]["error"]}), 500

    return jsonify([
        {
            "title": c.get('title', 'Unknown Career'),
            "match_score": round(c.get('match_score', 0), 1)
        }
        for c in careers[:3]  # top 3
    ])


@dashboard_bp.route('/career_path')
def get_career_path():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    title = request.args.get('title')
    careers = match_careers(user_id)

    # Handle RAG errors
    if isinstance(careers, list) and careers and "error" in careers[0]:
        return jsonify({"error": careers[0]["error"]}), 500

    if title:
        career = next((c for c in careers if c.get('title', '').lower() == title.lower()), None)
        if not career:
            return jsonify({"error": "Career not found"}), 404
        return jsonify({
            "title": career['title'],
            "roadmap": career.get('roadmap', 'Not available'),
            "timeline": career.get('timeline', 'Not available'),
            "skills": career.get('skills', 'N/A'),
            "rationale": career.get('rationale', 'Not available'),
            "match_score": round(career.get('match_score', 0), 1)
        })

    # Return top 3 for explorer
    return jsonify([
        {
            "title": c.get('title', 'Unknown Career'),
            "rationale": c.get('rationale', 'Not available'),
            "roadmap": c.get('roadmap', 'Not available'),
            "timeline": c.get('timeline', 'Not available'),
            "skills": c.get('skills', 'N/A'),
            "match_score": round(c.get('match_score', 0), 1)
        }
        for c in careers[:3]  # Top 3 careers
    ])
