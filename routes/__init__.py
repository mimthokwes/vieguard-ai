from flask import Flask, Blueprint, jsonify
from modules.RAG.controller import rag_bp, chatbot_bp

# Centralized API Blueprint with prefix '/api'
api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/", methods=["GET"])
def api_index():
    """Returns overview of all available AI endpoints."""
    return jsonify({
        "status": "success",
        "service": "VieGuard AI Backend",
        "version": "1.1.0",
        "modules": [
            {
                "name": "Chatbot RAG (Google Gemini + Vector Index)",
                "prefixes": ["/api/chatbot", "/api/rag"],
                "endpoints": [
                    {"method": "GET", "path": "/api/chatbot/model", "description": "List all available Gemini models"},
                    {"method": "POST", "path": "/api/chatbot/chat", "description": "Ask question to RAG chatbot"},
                    {"method": "POST", "path": "/api/chatbot/build-index", "description": "Rebuild vector index from knowledge/ folder"},
                    {"method": "GET", "path": "/api/chatbot/health", "description": "Check knowledge files & index status"},
                    {"method": "POST", "path": "/api/chatbot/documents", "description": "Upload new document to knowledge/"}
                ]
            }
        ]
    }), 200


def register_routes(app: Flask) -> None:
    """
    Register all modular blueprints to the main Flask application.
    Adding a new module (e.g. regression, classification) only requires
    registering its blueprint here.
    """
    # 1. RAG and Chatbot blueprints
    api_bp.register_blueprint(rag_bp)
    api_bp.register_blueprint(chatbot_bp)

    # 2. Register main API blueprint to Flask app
    app.register_blueprint(api_bp)
