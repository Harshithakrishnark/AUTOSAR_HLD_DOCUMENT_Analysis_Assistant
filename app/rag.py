"""Optional Ollama answer generation grounded in retrieved page excerpts."""
import requests


def ask_ollama(question: str, evidence: list[dict], model: str = "llama3.2:3b") -> str:
    if not evidence:
        return "I could not find matching evidence in the uploaded document. Try different keywords."
    context = "\n\n".join(
        f"[Page {item['page']}] {item['excerpt']}" for item in evidence
    )
    prompt = f"""You are an assistant for reviewing automotive HLD documents.
Answer only using the supplied evidence. If the evidence is insufficient, say so.
Cite claims with page references such as [Page 3]. Do not invent AUTOSAR requirements.

Evidence:
{context}

Question: {question}
Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()
