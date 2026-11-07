from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class RAGQueryRequest(BaseModel):
    # Support both 'question' (like reference project) and 'query'
    question: Optional[str] = Field(None, description="Pertanyaan untuk chatbot RAG")
    query: Optional[str] = Field(None, description="Alias pertanyaan untuk chatbot RAG")
    top_k: int = Field(default=3, ge=1, le=10, description="Jumlah referensi konteks yang diambil")

    @model_validator(mode="before")
    @classmethod
    def check_question_or_query(cls, values):
        if isinstance(values, dict):
            q = values.get("question") or values.get("query")
            if not q or not str(q).strip():
                raise ValueError("Pertanyaan tidak boleh kosong (isi 'question' atau 'query').")
            values["question"] = str(q).strip()
            values["query"] = str(q).strip()
        return values


class SourceMatch(BaseModel):
    id: int
    text: str
    source: Optional[str] = None
    score: float


class RAGQueryResponse(BaseModel):
    status: str = "success"
    question: str
    answer: str
    sources: List[SourceMatch]
    model_used: str
    execution_time_ms: float


class ModelInfo(BaseModel):
    name: str
    clean_name: str
    supported_actions: List[str] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    status: str = "success"
    current_chat_model: str
    current_embedding_model: str
    total_models: int
    models: List[ModelInfo]


class IndexBuildResponse(BaseModel):
    status: str = "success"
    message: str
    total_chunks: int
    dimension: int
    sources: List[str]
    embedding_model: str


class DocumentAddRequest(BaseModel):
    filename: str = Field(..., min_length=1, description="Nama file (contoh: sop_tambahan.txt)")
    content: str = Field(..., min_length=1, description="Isi teks panduan/dokumen")
    rebuild_index: bool = Field(default=True, description="Apakah langsung memperbarui index vektor")
