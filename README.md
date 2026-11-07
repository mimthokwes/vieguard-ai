# VieGuard AI Backend (Flask & RAG Gemini)

Backend AI modular terisolasi untuk VieGuard yang mengimplementasikan arsitektur RAG (Retrieval-Augmented Generation) berbasis Google GenAI SDK (`google-genai`).

---

## 📁 Struktur Direktori

```text
AI-vie/
├── main.py                     # Entry point Flask & CORS config
├── build_index.py              # Script CLI untuk membangun index vektor dari knowledge/
├── check_models.py             # Script CLI untuk melihat daftar model Gemini yang tersedia
├── requirements.txt            # Dependensi (google-genai, numpy, flask, pydantic, dll.)
├── .env                        # Konfigurasi GEMINI_API_KEY, model, dan port
├── .env.example
├── knowledge/                  # 📂 Folder untuk menaruh dokumen panduan (.txt / .md)
│   └── panduan_vieguard.txt
├── index/                      # 📂 Folder output vector index (dihasilkan otomatis)
│   ├── chunks.json             # Metadata & teks potongan dokumen
│   └── embeddings.npy          # Matrix vektor embedding numpy
├── routes/
│   └── __init__.py             # Router registry: menyatukan seluruh blueprint modul AI
└── modules/
    ├── __init__.py
    └── RAG/
        ├── __init__.py
        ├── types.py            # Pydantic schemas (validasi request, response, model info)
        ├── service.py          # Logika RAG: cosine similarity, embedding, Gemini generation
        └── controller.py       # Blueprint controller (/api/chatbot/... & /api/rag/...)
```

---

## ⚙️ Konfigurasi Environment (`.env`)

File `.env` sudah dikonfigurasi dengan API Key Gemini Anda:
```ini
PORT=5001
HOST=0.0.0.0
DEBUG=True

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
```

---

## 🛠️ CLI Utilities (Terminal)

1. **Melihat Model Gemini yang Tersedia:**
   ```bash
   python check_models.py
   ```

2. **Membangun Ulang Vector Index dari `knowledge/`:**
   ```bash
   python build_index.py
   ```
   *Cukup masukkan file `.txt` atau `.md` baru ke folder `knowledge/` lalu jalankan perintah ini.*

---

## 🚀 Menjalankan Server API

```bash
cd /home/mimthokwes/coding/tugas-kuliah/project/vieguard/AI-vie
source venv/bin/activate
python main.py
```
Server berjalan di `http://0.0.0.0:5001`.

---

## 🔌 Daftar Endpoint REST API

Semua endpoint dapat diakses melalui prefix `/api/chatbot` maupun `/api/rag`:

### 1. Cek List Model Gemini (Untuk Developing)
- **Method**: `GET`
- **URL**: `http://localhost:5001/api/chatbot/model`
- **Response**: Menampilkan 26 model Gemini beserta *supported actions* (`generateContent`, `embedContent`, dll).

### 2. Chat / Tanya RAG
- **Method**: `POST`
- **URL**: `http://localhost:5001/api/chatbot/chat` *(atau `/api/rag/chat`)*
- **Body JSON**:
  ```json
  {
    "question": "Bagaimana sistem enkripsi transmisi sensor di VieGuard?",
    "top_k": 3
  }
  ```
- **Response JSON**:
  ```json
  {
    "status": "success",
    "question": "Bagaimana sistem enkripsi transmisi sensor di VieGuard?",
    "answer": "Seluruh transmisi data antara sensor IoT, mobile app, dan server pusat diamankan menggunakan TLS 1.3 dan enkripsi AES-256-GCM [D3].",
    "sources": [
      {
        "id": 3,
        "score": 0.812,
        "source": "panduan_vieguard.txt",
        "text": "PROTOKOL KEAMANAN DAN ENKRIPSI..."
      }
    ],
    "source_ids": ["D3"],
    "model_used": "gemini-3.5-flash-lite",
    "execution_time_ms": 780.4
  }
  ```

### 3. Rebuild Index secara Remote via API
- **Method**: `POST`
- **URL**: `http://localhost:5001/api/chatbot/build-index`
- **Response**: Membangun ulang vektor index dari dokumen di `knowledge/`.

### 4. Cek Health & Status Knowledge Base
- **Method**: `GET`
- **URL**: `http://localhost:5001/api/chatbot/health`
- **Response**: Menampilkan daftar file dalam `knowledge/`, status index, dan jumlah chunk.

### 5. Upload Dokumen Baru via API
- **Method**: `POST`
- **URL**: `http://localhost:5001/api/chatbot/documents`
- **Body JSON**:
  ```json
  {
    "filename": "sop_patroli_baru.txt",
    "content": "Jadwal patroli malam dimulai pukul 22.00 WIB...",
    "rebuild_index": true
  }
  ```
