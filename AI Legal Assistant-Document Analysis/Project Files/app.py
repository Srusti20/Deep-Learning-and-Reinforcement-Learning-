import streamlit as st
import pandas as pd
from transformers import pipeline
from fpdf import FPDF
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import plotly.graph_objects as go
from pdf_handler import render_clause_type_pie
# Custom PDF utilities
from pdf_handler import process_pdf, display_pdf_with_annotations, initialize_session_state

@st.cache_data
def load_csv_sample():
    df = pd.read_csv("cuad_descriptions.csv", encoding="utf-8-sig")
    return df

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="t5-base", tokenizer="t5-base")

def split_into_chunks(text, max_tokens=500):
    words = text.split()
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
def remove_non_latin1(text):
    # Removes characters that can't be encoded in latin-1 (e.g., emojis)
    return re.sub(r'[^\x00-\xFF]', '', text)
def generate_pdf(summary_text):
    cleaned_summary = remove_non_latin1(summary_text)
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Legal Document Summary", ln=True, align="C")
    pdf.ln(8)

    # Section: Structured Information
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Structured Information Extracted:", ln=True)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    lines = cleaned_summary.strip().split('\n')

    for line in lines:
        if line.startswith("-  "):
            key_val = line[3:].split(":", 1)
            if len(key_val) == 2:
                key, val = key_val
                pdf.set_font("Arial", "B", 12)
                pdf.cell(50, 8, f"{key.strip()}:", ln=0)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 8, val.strip())
        elif line.strip() == "---":
            pdf.ln(4)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

    # Section: Clause Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Clause Summary:", ln=True)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    body_lines = [line for line in lines if not line.startswith("-  ") and line.strip() != "---" and not line.strip().lower().startswith("structured")]
    body = " ".join(body_lines).strip()

    if body:
        for paragraph in re.split(r'\.\s+', body):
            if paragraph:
                pdf.multi_cell(0, 8, paragraph.strip() + ".", align="J")

    # Output
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name


def extract_key_clauses(text, keywords=None):
    if keywords is None:
        keywords = [
        "liability", "termination", "confidentiality", "obligation", "dispute",
        "indemnification", "governing law", "force majeure", "payment", "warranty", "arbitration"
    ]
    clauses = []
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            clauses.append(sentence.strip())
    return clauses

def initialize_session_state():
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'page_texts' not in st.session_state:
        st.session_state.page_texts = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'key_clauses' not in st.session_state:
        st.session_state.key_clauses = []

def extract_between(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Not clearly stated"

def risk_compliance_analysis(text: str):
    findings = []
    risk_score = 0

    if "governing law" not in text.lower():
        findings.append("⚠ No Governing Law clause found.")
        risk_score += 2

    if "liability" in text.lower() and not re.search(r"(cap|limit|maximum|not exceed|shall not exceed)", text.lower()):
        findings.append("⚠ Uncapped Liability clause detected.")
        risk_score += 3

    if "termination" not in text.lower():
        findings.append("⚠ No Termination clause found.")
        risk_score += 2

    if "indemnification" not in text.lower():
        findings.append("⚠ Missing Indemnification clause.")
        risk_score += 2

    if "force majeure" not in text.lower():
        findings.append("⚠ Force Majeure clause not found.")
        risk_score += 1

    return findings, min(risk_score, 10)
clause_explanations = {
    "⚠ No Governing Law clause found.": "Specifies which state's or country's laws will govern the contract. Omitting it can lead to legal ambiguity.",
    "⚠ Uncapped Liability clause detected.": "Exposes a party to unlimited legal or financial risks. It's a major risk indicator.",
    "⚠ No Termination clause found.": "A termination clause defines how either party can exit the agreement legally.",
    "⚠ Missing Indemnification clause.": "Without it, parties may not be protected against third-party claims or losses.",
    "⚠ Force Majeure clause not found.": "This protects both parties from liability due to events beyond their control (e.g., natural disasters)."
}
initialize_session_state()
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("AI Legal Assistant — Document Analysis")
st.sidebar.title("📁 Explore Options")
doc_source = st.sidebar.radio("Select document source:", ["Explore Clause Dataset", "Upload and Analyze PDF"])

if doc_source == "Upload and Analyze PDF":
    uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type="pdf")

    if uploaded_file:
        if 'pdf_text' not in st.session_state or st.session_state.pdf_text == "":
            text, page_texts = process_pdf(uploaded_file)
            st.session_state.pdf_text = text
            st.session_state.page_texts = page_texts
            # Recompute clause type counts after setting pdf_text
            text = st.session_state.pdf_text.lower()
            clause_keywords = {
                "Confidentiality": ["confidential", "non-disclosure", "nda"],
                "Termination": ["terminate", "termination", "expiration"],
                "Arbitration": ["arbitration", "dispute resolution"],
                "Payment": ["payment", "fees", "compensation"],
                "Liability": ["liability", "liable", "responsibility"],
                "Governing Law": ["governing law", "jurisdiction"],
                "Force Majeure": ["force majeure", "acts of god"],
                "Indemnification": ["indemnify", "indemnification"],
                "Warranty": ["warranty", "guarantee"],
                "Intellectual Property": ["intellectual property", "IP rights", "ownership"],
                "Assignment": ["assign", "assignment", "transfer of rights"],
                "Amendment": ["amendment", "modification", "change"],
                "Notices": ["notice", "written notice", "communication"],
                "Severability": ["severability", "invalid provision"],
            }

            clause_type_counts = {}
            for clause, keywords in clause_keywords.items():
                clause_type_counts[clause] = sum(1 for word in keywords if word in text)

# Only store clauses that are found
            clause_type_counts = {k: v for k, v in clause_type_counts.items() if v > 0}
            st.session_state.clause_type_counts = clause_type_counts

            

        display_pdf_with_annotations(uploaded_file, st.session_state.annotations)
        full_text = st.session_state.pdf_text.lower()
        st.subheader("Key Clause Extraction")

        if st.button("Extract Key Clauses"):
            key_clauses = extract_key_clauses(full_text)
            if key_clauses:
                for i, clause in enumerate(key_clauses, 1):
                    st.markdown(f"**Clause {i}:** {clause}")
                st.session_state["key_clauses"] = key_clauses
            else:
                st.warning("No key clauses were identified based on the keywords.")
        st.subheader("Clause Presence Detection")
        expected_clauses = {
            "Confidentiality", "Termination", "Arbitration", "Payment",
            "Liability", "Governing Law", "Force Majeure", "Indemnification"
        }
        detected_clauses = set()

        for clause in expected_clauses:
            if clause.lower() in full_text:
                detected_clauses.add(clause)

        missing_clauses = expected_clauses - detected_clauses

        if missing_clauses:
            st.warning("⚠ The following important clauses were NOT found in the uploaded document:")
            for mc in missing_clauses:
                st.markdown(f"- {mc}")
        else:
            st.success("All expected clauses were found in the document!")
        

        st.subheader("Risk & Compliance Checker")
        issues, risk_score = risk_compliance_analysis(full_text)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Score (0 = Safe, 10 = Risky)"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "crimson" if risk_score > 6 else "green"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgreen"},
                    {'range': [4, 7], 'color': "yellow"},
                    {'range': [7, 10], 'color': "tomato"},
                ],
            }
        ))
        st.plotly_chart(fig)
        if risk_score >6:
            st.error("This document is potentially *fraudulent*. High risk score detected.")
        else:
            st.success("This document appears to be *trustworthy*. Risk score is low.")

        if issues:
            st.error("⚠ Compliance Findings:")
            for issue in issues:
                st.markdown(f"- {issue}")
                explanation = clause_explanations.get(issue.strip())
                if explanation:
                    st.caption(f"{explanation}")
                else:
                    st.caption("No detailed explanation available.")

        else:
            st.success("No major compliance risks detected.")

        render_clause_type_pie()
        if st.button("Summarize Document"):
            chunks = split_into_chunks(st.session_state.pdf_text)
            summarizer = get_summarizer()
            summaries = []

            for i, chunk in enumerate(chunks):
                with st.spinner(f"Summarizing part {i+1}..."):
                    result = summarizer("summarize: " + chunk, max_length=150, min_length=40, do_sample=False)
                    summaries.append(result[0]['summary_text'])

            full_summary = "\n".join(summaries)

            party_1 = extract_between(full_text, r"between\s+(.*?)\s+and")
            party_2 = extract_between(full_text, r"and\s+(.*?)\s+(agree|with|to|and|on)")
            subject = extract_between(full_text, r"(service|employment|consulting|license|partnership)\s+agreement")
            
            date = extract_between(full_text, r"on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})")
            law = extract_between(full_text, r"governed by.*?laws of\s+([A-Za-z\s,]+)")
            structured_info = f"""
*Structured Information Extracted:*
-  *Agreement Type*: {subject}
-  *Parties*: {party_1} and {party_2}
-  *Date*: {date}
-  *Governing Law*: {law}
"""
            
            st.session_state["summary"] = structured_info + "\n\n---\n\n" + full_summary
            st.success("Detailed Summary with Key Info Generated!")

    if "summary" in st.session_state and st.session_state["summary"].strip():
        st.subheader("Document Summary")
        st.text_area("Summary Output", st.session_state["summary"], height=300)

        pdf_file_path = generate_pdf(st.session_state["summary"])
        with open(pdf_file_path, "rb") as f:
            st.download_button(
                label="Download Summary as PDF",
                data=f.read(),
                file_name="Document_Summary.pdf",
                mime="application/pdf"
            )
else:
    df = load_csv_sample()
    st.subheader("Explore Clauses and Understand Their Meanings")
    st.dataframe(df.head())

    text_column = st.selectbox("Choose a Clause Category to Explore:", df.columns.tolist())
    row_idx = st.slider("Select Category Number", 0, len(df) - 1, 0)
    selected_text = str(df.iloc[row_idx][text_column])
    st.markdown("###  Selected Clause Text")
    st.write(selected_text)
