"""PDF extraction and evidence snippet helpers."""
from pathlib import Path
import fitz



def extract_pdf_pages(file_path):
    """Extract non-empty text from each PDF page."""
    pages = []

    if hasattr(file_path, "getvalue"):
        pdf_bytes = file_path.getvalue()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page": page_number, "text": text})
    else:
        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page": page_number, "text": text})

    return pages



def search_pages(pages: list[dict], query: str, limit: int = 5) -> list[dict]:
    """Return page-level matches using simple case-insensitive keyword overlap."""
    terms = {t.lower().strip(".,:;!?()[]{}") for t in query.split() if len(t) > 2}
    if not terms:
        return []
    scored = []
    for item in pages:
        text = item["text"]
        lower = text.lower()
        score = sum(lower.count(term) for term in terms)
        if score:
            # Show a compact excerpt around the first matching term.
            positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
            pos = min(positions) if positions else 0
            start = max(0, pos - 180)
            excerpt = " ".join(text[start:pos + 650].split())
            scored.append({"page": item["page"], "score": score, "excerpt": excerpt})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]
