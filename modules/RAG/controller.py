from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from modules.RAG.types import (
    RAGQueryRequest,
    DocumentAddRequest,
)
from modules.RAG.service import rag_service

# Define Blueprints for RAG and Chatbot
rag_bp = Blueprint("rag", __name__, url_prefix="/rag")
chatbot_bp = Blueprint("chatbot", __name__, url_prefix="/chatbot")


def handle_health():
    """Cek status modul RAG/Chatbot, daftar file knowledge, dan status indeks."""
    try:
        info = rag_service.list_knowledge()
        return jsonify(info), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def handle_list_models():
    """
    Endpoint untuk mengecek semua list model Gemini yang dapat digunakan.
    Sangat berguna untuk proses developing.
    """
    try:
        models_resp = rag_service.list_available_models()
        return jsonify(models_resp.model_dump()), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal mengambil daftar model dari Gemini API: {str(e)}"
        }), 500


def handle_build_index():
    """Membangun ulang vektor index dari seluruh file di folder knowledge/."""
    try:
        result = rag_service.build_index()
        return jsonify(result.model_dump()), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal membuat index: {str(e)}"
        }), 500


def handle_chat():
    """
    Endpoint RAG Chatbot:
    Menerima pertanyaan, melakukan retrieval dari index, dan mengembalikan jawaban Gemini.
    Mendukung format {'question': '...'} dan {'query': '...'}.
    """
    body = request.get_json(silent=True) or {}
    try:
        req_obj = RAGQueryRequest(**body)
    except (ValidationError, ValueError) as err:
        return jsonify({
            "status": "error",
            "message": "Input tidak valid. Harap sertakan field 'question' atau 'query'.",
            "errors": err.errors() if hasattr(err, "errors") else str(err)
        }), 400

    try:
        response = rag_service.answer_with_rag(req_obj)
        res_data = response.model_dump()
        res_data["source_ids"] = [f"D{s['id']}" for s in res_data["sources"]]
        return jsonify(res_data), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Proses RAG gagal: {str(e)}"
        }), 500


def handle_upload_file():
    """
    Menerima file upload (multipart/form-data: PDF, DOCX, TXT, MD) dari backend utama,
    menyimpannya ke folder knowledge/, dan langsung memicu build_index().
    """
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "File tidak ditemukan dalam request (pastikan form field bernama 'file')."
        }), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename or not uploaded_file.filename.strip():
        return jsonify({
            "status": "error",
            "message": "Nama file tidak boleh kosong."
        }), 400

    try:
        result = rag_service.save_uploaded_file(uploaded_file)
        return jsonify(result), 201
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal memproses dan mengindeks file: {str(e)}"
        }), 500


def handle_add_document_text():
    """Menambahkan konten dokumen teks ke folder knowledge/ (opsional/alternatif teks mentah)."""
    body = request.get_json(silent=True) or {}
    try:
        req_obj = DocumentAddRequest(**body)
    except ValidationError as err:
        return jsonify({
            "status": "error",
            "message": "Validasi dokumen gagal.",
            "errors": err.errors()
        }), 400

    try:
        result = rag_service.add_document(
            filename=req_obj.filename,
            content=req_obj.content,
            rebuild_index=req_obj.rebuild_index
        )
        return jsonify(result), 201
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal menyimpan dokumen: {str(e)}"
        }), 500


# --- Register routes on rag_bp (/api/rag/...) ---
rag_bp.add_url_rule("/", view_func=handle_health, methods=["GET"])
rag_bp.add_url_rule("/health", view_func=handle_health, methods=["GET"])
rag_bp.add_url_rule("/model", view_func=handle_list_models, methods=["GET"])
rag_bp.add_url_rule("/models", view_func=handle_list_models, methods=["GET"])
rag_bp.add_url_rule("/build-index", view_func=handle_build_index, methods=["POST"])
rag_bp.add_url_rule("/chat", view_func=handle_chat, methods=["POST"])
rag_bp.add_url_rule("/query", view_func=handle_chat, methods=["POST"])
rag_bp.add_url_rule("/upload", view_func=handle_upload_file, methods=["POST"])
rag_bp.add_url_rule("/documents", view_func=handle_add_document_text, methods=["POST"])
rag_bp.add_url_rule("/documents", view_func=handle_health, methods=["GET"])

# --- Register routes on chatbot_bp (/api/chatbot/...) ---
chatbot_bp.add_url_rule("/", view_func=handle_health, methods=["GET"])
chatbot_bp.add_url_rule("/health", view_func=handle_health, methods=["GET"])
chatbot_bp.add_url_rule("/model", view_func=handle_list_models, methods=["GET"])
chatbot_bp.add_url_rule("/models", view_func=handle_list_models, methods=["GET"])
chatbot_bp.add_url_rule("/build-index", view_func=handle_build_index, methods=["POST"])
chatbot_bp.add_url_rule("/chat", view_func=handle_chat, methods=["POST"])
chatbot_bp.add_url_rule("/query", view_func=handle_chat, methods=["POST"])
chatbot_bp.add_url_rule("/upload", view_func=handle_upload_file, methods=["POST"])
chatbot_bp.add_url_rule("/documents", view_func=handle_add_document_text, methods=["POST"])
chatbot_bp.add_url_rule("/documents", view_func=handle_health, methods=["GET"])
