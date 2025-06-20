import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
from typing import List, Tuple
import pandas as pd
import plotly.express as px

def process_pdf(uploaded_file) -> Tuple[str, List[str]]:
    text = ""
    page_texts = []

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        doc = fitz.open(tmp_file_path)
        for page in doc:
            page_text = page.get_text()
            text += page_text
            page_texts.append(page_text)
        doc.close()
    finally:
        os.unlink(tmp_file_path)

    return text, page_texts

# 🔥 This function is now a dummy — preview & annotation fully removed
def display_pdf_with_annotations(*args, **kwargs):
    pass

def render_clause_type_pie():
    if 'clause_type_counts' in st.session_state and st.session_state.clause_type_counts:
        df_clauses = pd.DataFrame(list(st.session_state.clause_type_counts.items()), columns=["Clause Type", "Count"])
        st.markdown("### Clause Category Breakdown")
        fig = px.pie(df_clauses, names="Clause Type", values="Count", title="Clause Proportion")
        st.plotly_chart(fig)

def initialize_session_state():
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'page_texts' not in st.session_state:
        st.session_state.page_texts = []

    # Removed: annotations and current_page (not needed anymore)

    # Clause Classification Setup
    if 'clause_type_counts' not in st.session_state:
        text = st.session_state.pdf_text.lower() if 'pdf_text' in st.session_state else ""
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

        clause_type_counts = {k: v for k, v in clause_type_counts.items() if v > 0}
        st.session_state.clause_type_counts = clause_type_counts
