"""
app/__init__.py
Creates and configures the Flask application.
Registers all blueprints and sets up session management.
"""

from flask import Flask, redirect, url_for
from dotenv import load_dotenv
from datetime import timedelta
import os
import logging

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """
    Application factory function.
    Initializes Flask app, configures settings, and registers blueprints.
    """
    app = Flask(__name__)

    # === Configuration ===
    app.secret_key = os.getenv("FLASK_SECRET_KEY")
    if not app.secret_key:
        logger.warning("FLASK_SECRET_KEY not set! Using fallback (insecure for production).")
        app.secret_key = "fallback_secret_key_change_in_production"

    app.permanent_session_lifetime = timedelta(minutes=30)
    app.config['SESSION_PERMANENT'] = True
    app.config['SESSION_TYPE'] = 'filesystem'  # Optional: use redis in production

    # === Register Blueprints ===
    try:
        from app.routes.auth_routes import auth_bp
        from app.routes.career_routes import career_bp
        from app.routes.dashboard_routes import dashboard_bp

        app.register_blueprint(auth_bp)
        app.register_blueprint(career_bp)
        app.register_blueprint(dashboard_bp)

        logger.info("All blueprints registered successfully.")

    except ImportError as e:
        logger.error(f"Failed to import blueprint: {e}")
        raise

    # === Root Route ===
    @app.route("/")
    def root():
        """
        Redirect root URL to auth home page.
        """
        return redirect(url_for("auth_bp.home"))

    # === Error Handlers (Optional but Recommended) ===
    @app.errorhandler(404)
    def not_found(e):
        return redirect(url_for("auth_bp.home"))

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Server Error: {e}")
        return "Something went wrong. Please try again later.", 500

    return app