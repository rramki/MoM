import streamlit as st
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime
import tempfile
import os

st.set_page_config(page_title="Professional MoM Generator", layout="centered")
st.title("📄 Professional AI Minutes of Meeting Generator")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

uploaded_file = st.file_uploader("Upload Meeting Audio", type=["mp3", "wav", "m4a"])

if uploaded_file:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Transcription via OpenAI API
    with st.spinner("Transcribing audio..."):
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            ).text

    st.subheader("Transcript")
    st.write(transcript)

    # Generate structured MoM
    with st.spinner("Generating professional MoM..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional corporate meeting assistant."},
                {"role": "user", "content": f"""
                Convert this transcript into structured professional Minutes of Meeting including:

                - Meeting Title
                - Date
                - Participants
                - Agenda
                - Key Discussion Points
                - Decisions Taken
                - Action Items
                - Next Meeting Details

                Transcript:
                {transcript}
                """}
            ]
        )

        mom_text = response.choices[0].message.content

    st.subheader("Structured MoM")
    st.write(mom_text)

    # PDF Generation
    def generate_pdf(text):
        file_path = "Minutes_of_Meeting.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()

        elements = []
        elements.append(Paragraph("MINUTES OF MEETING", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y')}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(text.replace("\n", "<br/>"), styles["Normal"]))

        doc.build(elements)
        return file_path

    pdf_file = generate_pdf(mom_text)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📥 Download Professional MoM PDF",
            f,
            file_name="Minutes_of_Meeting.pdf",
            mime="application/pdf"
        )

    os.remove(tmp_path)
