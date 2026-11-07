#!/usr/bin/env python3
"""
Script utilitas untuk mengecek seluruh model Gemini yang tersedia pada API key Anda.
Dapat dijalankan langsung di terminal:
    python check_models.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env dari root direktori AI-vie
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY belum diisi di file .env")
    exit(1)

print(f"🔑 Memeriksa model dengan API Key: {api_key[:6]}...{api_key[-4:]}")

try:
    client = genai.Client(api_key=api_key)
    all_models = list(client.models.list())
    
    print(f"\n✅ Ditemukan {len(all_models)} model yang dapat diakses:\n")
    print(f"{'No':<4} | {'Nama Model':<35} | {'Supported Actions'}")
    print("-" * 80)

    embedding_models = []
    chat_models = []

    for i, m in enumerate(all_models, 1):
        clean_name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", [])
        actions_str = ", ".join(actions) if actions else "-"
        print(f"{i:<4} | {clean_name:<35} | {actions_str}")

        if "embedContent" in actions or "embed" in clean_name:
            embedding_models.append(clean_name)
        if "generateContent" in actions:
            chat_models.append(clean_name)

    print("\n" + "=" * 80)
    print("📌 REKOMENDASI UNTUK VIEGUARD RAG:")
    print("  • Chat / Generation Model : gemini-3.5-flash-lite  (Cepat, cerdas & stabil)")
    print("  • Embedding Model         : gemini-embedding-001  (Dimensi 3072, standar vector)")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Gagal memuat model: {e}")
    exit(1)
