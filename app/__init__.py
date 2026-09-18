import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.config["DATABASE_URL"] = os.getenv("DATABASE_URL")
    app.config["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    app.config["SQL_MODEL"] = os.getenv("SQL_MODEL", "qwen2.5-coder:7b")
    app.config["ROUTER_MODEL"] = os.getenv("ROUTER_MODEL", "llama3.2:3b")
    app.config["EMBEDDING_MODEL"] = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6MB request cap - a bit above ingest.py's 5MB file check, so that check (with a clearer message) fires first

    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app
