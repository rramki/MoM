import streamlit as st
import whisper
from transformers import pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.pagesizes import A4
import tempfile
import os

st.set_page_config(page_title="Minutes of Meeting Generator", layout="centered")

st.title("📄 AI Minutes of Meeting Generator")
st.write("Upload meeting audio file and download professional MoM PDF")

# Load models (cached)
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return whisper_model, summarizer

whisper_model, summarizer = load_models()

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Transcribing audio..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"]

    st.subheader("📝 Transcript")
    st.write(transcript)

    with st.spinner("Generating summary..."):
        summary = summarizer(transcript, max_length=300, min_length=80, do_sample=False)[0]["summary_text"]

    st.subheader("📌 Meeting Summary")
    st.write(summary)

    # Generate Professional PDF
    def generate_pdf(summary_text):
        file_path = "Minutes_of_Meeting.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()

        elements = []

        title_style = styles["Heading1"]
        normal_style = styles["Normal"]

        elements.append(Paragraph("MINUTES OF MEETING", title_style))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("Meeting Summary:", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(summary_text, normal_style))

        doc.build(elements)

        return file_path

    pdf_file = generate_pdf(summary)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download Minutes of Meeting PDF",
            data=f,
            file_name="Minutes_of_Meeting.pdf",
            mime="application/pdf"
        )

    os.remove(tmp_path)
