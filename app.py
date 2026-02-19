import streamlit as st
import whisper
from transformers import pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import tempfile
import os
from datetime import datetime

st.set_page_config(page_title="MoM Generator", layout="centered")
st.title("📄 AI Minutes of Meeting Generator")

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")  # smaller model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return whisper_model, summarizer

whisper_model, summarizer = load_models()

uploaded_file = st.file_uploader("Upload Meeting Audio", type=["mp3", "wav", "m4a"])

if uploaded_file:

    st.audio(uploaded_file)

    with st.spinner("Transcribing..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"]

    st.subheader("Transcript")
    st.write(transcript)

    # 🔹 Break large transcript into chunks (prevents crash)
    def chunk_text(text, max_chunk=1000):
        return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    with st.spinner("Generating Summary..."):
        chunks = chunk_text(transcript)
        summary_parts = []

        for chunk in chunks[:3]:  # limit chunks to avoid memory issue
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
            summary_parts.append(summary)

        final_summary = " ".join(summary_parts)

    st.subheader("Structured Minutes")
    st.write(final_summary)

    # PDF generation
    def generate_pdf(text):
        file_path = "Minutes_of_Meeting.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()

        elements = []
        elements.append(Paragraph("MINUTES OF MEETING", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Date: {datetime.now().strftime('%d %B %Y')}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(text, styles["Normal"]))

        doc.build(elements)
        return file_path

    pdf_file = generate_pdf(final_summary)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "Download MoM PDF",
            f,
            file_name="Minutes_of_Meeting.pdf",
            mime="application/pdf"
        )

    os.remove(tmp_path)
