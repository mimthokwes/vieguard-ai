#!/usr/bin/env python3
"""
Script untuk membangun vector index dari seluruh file di folder knowledge/
Mendukung format: .txt, .md, .pdf, .docx, .doc
Dapat dijalankan langsung di terminal:
    python build_index.py
"""

import sys
from pathlib import Path

# Add current dir to path to reuse service
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from modules.RAG.service import rag_service

if __name__ == "__main__":
    print("🔄 Membaca file di folder knowledge/ dan membangun vector index...")
    try:
        res = rag_service.build_index()
        print("\n✅ INDEX BERHASIL DIPERBARUI!")
        print(f"  • Total chunks   : {res.total_chunks}")
        print(f"  • Vektor dimensi : {res.dimension}")
        print(f"  • Sumber file    : {', '.join(res.sources)}")
        print(f"  • Embedding Model: {res.embedding_model}")
    except Exception as e:
        print(f"\n❌ Gagal membangun index: {e}")
        sys.exit(1)
