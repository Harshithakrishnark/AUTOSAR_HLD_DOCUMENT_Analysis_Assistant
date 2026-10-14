from pathlib import Path
import tempfile
import streamlit as st
import joblib 
import pandas as pd

from document_utils import extract_pdf_pages, search_pages
from rag import ask_ollama

st.set_page_config(page_title="AUTOSAR HLD Assistant", page_icon="🚗", layout="wide")
st.title("AUTOSAR HLD Document Analysis Assistant")
st.caption("Upload an HLD PDF, search its contents, and review page-referenced evidence.")

with st.sidebar:
    st.header("Project information")
    st.write("Case Study 1 — AUTOSAR HLD Document Analysis Assistant")
    st.info("Pilot tool only. All findings require engineering review.")
    llm_model = st.text_input("Ollama model", value="llama3.2:3b")

uploaded = st.file_uploader("Upload an HLD PDF", type=["pdf"])
if uploaded:
    if uploaded.size > 30 * 1024 * 1024:
        st.error("The PDF is larger than 30 MB. Please upload a smaller file.")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = Path(tmp.name)
    try:
        pages = extract_pdf_pages(temp_path)
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
        st.stop()
    finally:
        temp_path.unlink(missing_ok=True)

    if not pages:
        st.warning("No selectable text was found. This may be a scanned PDF; OCR is not included yet.")
        st.stop()

    st.success(f"Extracted text from {len(pages)} pages.")
    total_chars = sum(len(p["text"]) for p in pages)
    c1, c2 = st.columns(2)
    c1.metric("Pages with text", len(pages))
    c2.metric("Extracted characters", f"{total_chars:,}")

    question = st.text_input(
        "Search or ask a question",
        placeholder="e.g., Which components and interfaces are described?"
    )
    if question:
        evidence = search_pages(pages, question, limit=5)
        if not evidence:
            st.warning("No matching page excerpts found. Try shorter or different keywords.")
        else:
            st.subheader("Retrieved evidence")
            for item in evidence:
                with st.expander(f"Page {item['page']} · match score {item['score']}", expanded=True):
                    st.write(item["excerpt"])

            if st.button("Generate answer with local Ollama model"):
                try:
                    with st.spinner("Generating a grounded answer..."):
                        answer = ask_ollama(question, evidence, llm_model)
                    st.subheader("AI answer")
                    st.write(answer)
                except Exception as exc:
                    st.error(
                        "Ollama could not generate an answer. Check that Ollama is installed, "
                        "running, and that the selected model is downloaded. "
                        f"Details: {exc}"
                    )

    with st.expander("Preview extracted pages"):
        for item in pages[:10]:
            st.markdown(f"**Page {item['page']}**")
            st.text(item["text"][:2500])
            st.divider()
else:
    st.markdown("""
    ### What this pilot supports
    - PDF text extraction with page numbers
    - Keyword-based evidence retrieval
    - Optional local LLM responses grounded in retrieved excerpts
    - A separate baseline ML classifier for document-section categories

    Upload a text-based HLD PDF to begin.
    """)


st.divider()
st.header("ML-Based HLD Section Classification")

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "section_classifier.joblib"
)

if model_path.exists():
    try:
        classifier = joblib.load(model_path)

        section_text = st.text_area(
            "Enter HLD section text",
            placeholder="Paste an HLD requirement or design description here..."
        )

        if st.button("Classify Section"):
            if section_text.strip():
                prediction = classifier.predict([section_text])[0]
                st.success(f"Predicted category: {prediction}")
            else:
                st.warning("Please enter some text first.")

    except Exception as exc:
        st.error(f"Could not load the classifier: {exc}")
else:
    st.warning("Train the ML classifier before using this feature.")


# --------------------------------------------------
# Model Evaluation Dashboard
# --------------------------------------------------

st.divider()
st.header("Model Evaluation Dashboard")

project_dir = Path(__file__).resolve().parent.parent
reports_dir = project_dir / "reports"

comparison_file = reports_dir / "model_comparison.csv"
matrix_file = reports_dir / "confusion_matrix.png"
classification_file = reports_dir / "classification_report.txt"

if comparison_file.exists():
    try:
        results_df = pd.read_csv(comparison_file)

        st.subheader("Model Performance Comparison")
        st.dataframe(results_df, use_container_width=True)

        if "model" in results_df.columns and "accuracy" in results_df.columns:
            st.subheader("Accuracy Comparison")

            chart_data = results_df.set_index("model")[["accuracy"]]
            st.bar_chart(chart_data)

        if "weighted_precision" in results_df.columns:
            st.subheader("Precision Comparison")
            st.bar_chart(
                results_df.set_index("model")[["weighted_precision"]]
            )

        if "weighted_recall" in results_df.columns:
            st.subheader("Recall Comparison")
            st.bar_chart(
                results_df.set_index("model")[["weighted_recall"]]
            )

        if "weighted_f1" in results_df.columns:
            st.subheader("F1-Score Comparison")
            st.bar_chart(
                results_df.set_index("model")[["weighted_f1"]]
            )

    except Exception as exc:
        st.error(f"Could not load model comparison: {exc}")
else:
    st.warning("Model comparison report not found. Train the models first.")

if matrix_file.exists():
    st.subheader("Confusion Matrix")
    st.image(str(matrix_file), use_container_width=True)

if classification_file.exists():
    st.subheader("Classification Report")
    report_text = classification_file.read_text(encoding="utf-8")
    st.code(report_text, language="text")



# --------------------------------------------------
# Document Comparison and Export
# --------------------------------------------------

st.divider()
st.header("Document Comparison and Export")

comparison_file = st.file_uploader(
    "Upload a second HLD PDF to compare",
    type=["pdf"],
    key="comparison_pdf"
)

if uploaded is not None and comparison_file is not None:
    try:
        uploaded.seek(0)
        first_pages = extract_pdf_pages(uploaded)
        comparison_file.seek(0)
        second_pages = extract_pdf_pages(comparison_file)

        first_text = "\n".join(
            page["text"] for page in first_pages
        )
        second_text = "\n".join(
            page["text"] for page in second_pages
        )

        st.subheader("Comparison Summary")

        first_words = set(first_text.lower().split())
        second_words = set(second_text.lower().split())

        common_words = first_words & second_words
        only_first = first_words - second_words
        only_second = second_words - first_words

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Words in first document", len(first_words))
            st.metric("Words unique to first", len(only_first))

        with col2:
            st.metric("Words in second document", len(second_words))
            st.metric("Words unique to second", len(only_second))

        st.metric("Common distinct words", len(common_words))

        findings = (
            "AUTOSAR HLD Document Comparison\n\n"
            f"First document: {uploaded.name}\n"
            f"Second document: {comparison_file.name}\n\n"
            f"Distinct words in first: {len(first_words)}\n"
            f"Distinct words in second: {len(second_words)}\n"
            f"Common distinct words: {len(common_words)}\n\n"
            "Words found only in the first document:\n"
            + ", ".join(sorted(only_first))
            + "\n\nWords found only in the second document:\n"
            + ", ".join(sorted(only_second))
        )

        st.subheader("Export Findings")

        st.download_button(
            "Download comparison report",
            data=findings,
            file_name="autosar_comparison_report.txt",
            mime="text/plain"
        )

    except Exception as exc:
        st.error(f"Could not compare documents: {exc}")

elif uploaded is None:
    st.info("Upload the first HLD PDF above to begin comparison.")
else:
    st.info("Upload a second HLD PDF to compare documents.")
