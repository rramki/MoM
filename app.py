import streamlit as st
import whisper
from transformers import pipeline
from fpdf import FPDF
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Minutes of Meeting Generator", layout="wide")

st.title("📄 AI Powered Minutes of Meeting Generator")

# Upload audio
audio_file = st.file_uploader("Upload Meeting Audio File", type=["mp3", "wav", "m4a"])

# Upload company logo
logo_file = st.file_uploader("Upload Company Logo (Optional)", type=["png", "jpg", "jpeg"])

company_name = st.text_input("Enter Company Name")
meeting_title = st.text_input("Enter Meeting Title")
meeting_date = st.date_input("Meeting Date")

if st.button("Generate Minutes of Meeting"):

    if audio_file is None:
        st.error("Please upload audio file.")
    else:
        with st.spinner("Transcribing Audio..."):
            # Save audio temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            transcript = result["text"]

        st.success("Transcription Completed!")

        # Summarization
        with st.spinner("Generating Professional Summary..."):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(transcript[:1024], max_length=300, min_length=100, do_sample=False)
            meeting_summary = summary[0]['summary_text']

        st.success("Summary Generated!")

        # Generate Professional MoM Format
        mom_text = f"""
Company Name: {company_name}
Meeting Title: {meeting_title}
Date: {meeting_date}

--- Meeting Summary ---

{meeting_summary}

--- Key Discussion Points ---
• {meeting_summary}

--- Action Items ---
• To be derived from discussion

--- Conclusion ---
Meeting concluded successfully.
"""

        # PDF Generation
        with st.spinner("Generating PDF..."):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Add Logo
            if logo_file is not None:
                logo = Image.open(logo_file)
                logo_path = "logo.png"
                logo.save(logo_path)
                pdf.image(logo_path, x=10, y=8, w=30)

            pdf.ln(40)

            for line in mom_text.split("\n"):
                pdf.multi_cell(0, 8, line)

            pdf_file = "Minutes_of_Meeting.pdf"
            pdf.output(pdf_file)

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download Minutes of Meeting PDF",
                data=f,
                file_name=pdf_file,
                mime="application/pdf"
            )

        st.success("PDF Generated Successfully!")
