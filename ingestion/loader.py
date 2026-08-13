"""
Loads raw documents (PDF, DOCX, TXT, MD) from a folder and extracts clean text.
"""
import os
import re
from pypdf import PdfReader
import docx


def _clean_text(text: str) -> str:
    """Basic cleaning: collapse whitespace, strip odd control chars."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return _clean_text("\n".join(pages))


def load_docx(path: str) -> str:
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    return _clean_text("\n".join(paragraphs))


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _clean_text(f.read())


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".md": load_txt,
}


def load_documents(folder: str) -> list[dict]:
    """
    Walks `folder`, extracts text from every supported file.
    Returns a list of {"source": filename, "text": full_text}.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    documents = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        loader = LOADERS.get(ext)
        if not loader:
            continue
        path = os.path.join(folder, filename)
        try:
            text = loader(path)
        except Exception as e:
            print(f"[loader] Skipping {filename}: {e}")
            continue
        if text.strip():
            documents.append({"source": filename, "text": text})
    return documents
