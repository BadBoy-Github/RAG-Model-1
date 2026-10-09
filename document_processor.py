import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pypdf

from config import CACHE_DIR, DEFAULT_EMBEDDING_MODEL, PDF_FILE_PATH, get_gemini_client


@dataclass
class DocumentChunk:
    chunk_id: int
    page: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            page=data["page"],
            text=data["text"],
        )


class PDFDocumentProcessor:
    """
    Handles PDF loading, text extraction, intelligent chunking,
    embedding generation with local caching, and similarity/keyword retrieval.
    """

    def __init__(
        self,
        pdf_path: Path = PDF_FILE_PATH,
        cache_dir: Path = CACHE_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ):
        self.pdf_path = Path(pdf_path)
        self.cache_dir = Path(cache_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.pages: List[str] = []
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._is_indexed = False

    def get_file_hash(self) -> str:
        """Compute MD5 hash of the PDF file to validate cache freshness."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")
        hasher = hashlib.md5()
        with open(self.pdf_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_pdf(self) -> List[str]:
        """Extract text from each page of the PDF document."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF document not found: {self.pdf_path}")

        reader = pypdf.PdfReader(str(self.pdf_path))
        self.pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean up whitespace while preserving line structure
            cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
            self.pages.append("\n".join(cleaned_lines))

        return self.pages

    def chunk_document(self) -> List[DocumentChunk]:
        """
        Split document pages into overlapping text chunks with page tracking.
        Designed to keep menu items and their prices/sections cohesive.
        """
        if not self.pages:
            self.load_pdf()

        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for page_idx, page_text in enumerate(self.pages):
            page_num = page_idx + 1
            lines = page_text.splitlines()

            # Group lines into chunks by character count
            current_chunk_lines: List[str] = []
            current_len = 0

            for line in lines:
                line_len = len(line) + 1
                if current_len + line_len > self.chunk_size and current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_idx,
                            page=page_num,
                            text=chunk_text,
                        )
                    )
                    chunk_idx += 1

                    # Retain overlap lines for context continuity
                    overlap_lines: List[str] = []
                    overlap_len = 0
                    for prev_line in reversed(current_chunk_lines):
                        if overlap_len + len(prev_line) < self.chunk_overlap:
                            overlap_lines.insert(0, prev_line)
                            overlap_len += len(prev_line) + 1
                        else:
                            break

                    current_chunk_lines = overlap_lines + [line]
                    current_len = sum(len(l) + 1 for l in current_chunk_lines)
                else:
                    current_chunk_lines.append(line)
                    current_len += line_len

            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_idx,
                        page=page_num,
                        text=chunk_text,
                    )
                )
                chunk_idx += 1

        self.chunks = chunks
        return chunks

    def _get_cache_filepath(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / "embeddings_cache.json"

    def _load_cache(self, current_hash: str) -> bool:
        """Attempt to load chunks and embeddings from local disk cache."""
        cache_file = self._get_cache_filepath()
        if not cache_file.exists():
            return False

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("file_hash") != current_hash:
                return False
            if data.get("embedding_model") != self.embedding_model:
                return False

            self.chunks = [DocumentChunk.from_dict(c) for c in data["chunks"]]
            self.embeddings = np.array(data["embeddings"], dtype=np.float32)
            self._is_indexed = True
            return True
        except Exception as e:
            print(f"[Warning] Failed to load cache: {e}. Recomputing embeddings.")
            return False

    def _save_cache(self, current_hash: str):
        """Save chunks and computed embeddings to local disk cache."""
        cache_file = self._get_cache_filepath()
        try:
            data = {
                "file_hash": current_hash,
                "embedding_model": self.embedding_model,
                "chunks": [c.to_dict() for c in self.chunks],
                "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save embeddings cache: {e}")

    def build_index(self, force_reindex: bool = False, verbose: bool = True) -> int:
        """
        Build the vector index.
        Uses cached embeddings if valid, or generates embeddings via Gemini API.
        """
        if not self.pages:
            self.load_pdf()

        file_hash = self.get_file_hash()

        if not force_reindex and self._load_cache(file_hash):
            if verbose:
                print(f"[RAG Index] Loaded {len(self.chunks)} chunks from local cache.")
            return len(self.chunks)

        if not self.chunks:
            self.chunk_document()

        if verbose:
            print(f"[RAG Index] Generating Gemini embeddings for {len(self.chunks)} chunks...")

        client = get_gemini_client()
        embeddings_list: List[List[float]] = []

        def _embed_batch(texts: List[str]):
            models_to_try = [self.embedding_model, "gemini-embedding-001", "gemini-embedding-2"]
            last_err = None
            for model_name in dict.fromkeys(models_to_try):
                try:
                    resp = client.models.embed_content(
                        model=model_name,
                        contents=texts,
                    )
                    self.embedding_model = model_name
                    return resp
                except Exception as ex:
                    last_err = ex
                    if "404" in str(ex) or "not found" in str(ex).lower():
                        continue
                    raise ex
            raise last_err

        # Batch embed contents to optimize network calls
        batch_size = 15
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            contents = [c.text for c in batch]
            response = _embed_batch(contents)
            for emb in response.embeddings:
                embeddings_list.append(emb.values)

        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        self._save_cache(file_hash)
        self._is_indexed = True

        if verbose:
            print(f"[RAG Index] Successfully indexed {len(self.chunks)} chunks (using {self.embedding_model}).")
        return len(self.chunks)

    def semantic_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Find top-k chunks most semantically similar to query using cosine similarity.
        """
        if not self._is_indexed or self.embeddings is None:
            self.build_index(verbose=False)

        client = get_gemini_client()
        try:
            query_resp = client.models.embed_content(
                model=self.embedding_model,
                contents=query,
            )
        except Exception:
            # Fallback model attempt
            query_resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=query,
            )
        query_vec = np.array(query_resp.embeddings[0].values, dtype=np.float32)

        # Compute cosine similarity
        norm_query = np.linalg.norm(query_vec)
        norm_chunks = np.linalg.norm(self.embeddings, axis=1)

        # Guard against zero norm
        denominator = norm_chunks * norm_query
        denominator = np.where(denominator == 0, 1e-10, denominator)
        scores = np.dot(self.embeddings, query_vec) / denominator

        # Top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "score": float(scores[idx]),
                    "text": chunk.text,
                }
            )
        return results

    def keyword_search(self, keyword: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search chunks using token and substring matching (no API call required).
        """
        if not self.chunks:
            self.chunk_document()

        kw_lower = keyword.strip().lower()
        terms = [t for t in kw_lower.split() if len(t) > 2]
        if not terms:
            terms = [kw_lower]

        scored_results: List[Tuple[float, DocumentChunk]] = []

        for chunk in self.chunks:
            text_lower = chunk.text.lower()
            score = 0.0

            # Exact phrase match bonus
            if kw_lower in text_lower:
                score += 3.0

            # Individual term matches
            for term in terms:
                count = text_lower.count(term)
                score += count * 1.0

            if score > 0:
                scored_results.append((score, chunk))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored_results[:top_k]:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "score": round(score, 2),
                    "text": chunk.text,
                }
            )
        return results

    def get_page_content(self, page_number: int) -> str:
        """Retrieve full text of a specific 1-indexed page."""
        if not self.pages:
            self.load_pdf()

        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]
        return f"Error: Page number {page_number} is out of bounds. The document has {len(self.pages)} pages."

    def get_document_summary(self) -> Dict[str, Any]:
        """Get structural overview of the PDF document."""
        if not self.pages:
            self.load_pdf()

        return {
            "total_pages": len(self.pages),
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "pages_overview": [
                {
                    "page": i + 1,
                    "char_count": len(page_text),
                    "first_line": page_text.splitlines()[0] if page_text.splitlines() else "",
                }
                for i, page_text in enumerate(self.pages)
            ],
        }
