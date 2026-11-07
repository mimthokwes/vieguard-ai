import os
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

from routes import register_routes

# Load environment variables from .env file
load_dotenv()


def create_app() -> Flask:
    """Application factory for VieGuard AI Backend."""
    app = Flask(__name__)

    # Enable Cross-Origin Resource Sharing (CORS) for web and mobile clients
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register all modular routes (RAG, and future AI models)
    register_routes(app)

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({
            "status": "online",
            "service": "VieGuard AI Service",
            "message": "AI backend is running and isolated from other services.",
            "api_documentation": "/api"
        }), 200

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "code": 404,
            "message": "Endpoint tidak ditemukan. Cek /api untuk daftar endpoint yang tersedia."
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "status": "error",
            "code": 500,
            "message": "Terjadi kesalahan internal pada server AI."
        }), 500

    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    print(f"🚀 Starting VieGuard AI Backend on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
