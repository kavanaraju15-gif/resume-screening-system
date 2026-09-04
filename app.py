import streamlit as st
import pandas as pd
import re

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="",
    layout="centered"
)


# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):
    text = str(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# TRAIN MODEL
# ==========================================

@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_csv("data/Resume.csv")

    # Clean resume text
    df["clean_resume"] = df["Resume_str"].apply(clean_text)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df["clean_resume"])

    # Target
    y = df["Category"]

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)

    model.fit(X, y)

    return model, vectorizer


model, vectorizer = train_model()


# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================
# DOCX TEXT EXTRACTION
# ==========================================

def extract_docx_text(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + "\n"

    return text


# ==========================================
# APPLICATION TITLE
# ==========================================

st.title(" Resume Screening System")

st.write(
    "Upload a resume or paste resume text "
    "and compare it with a job description."
)

st.divider()


# ==========================================
# RESUME UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    " Upload Resume",
    type=["pdf", "docx"]
)


# ==========================================
# PASTE RESUME
# ==========================================

resume_text = st.text_area(
    "Or Paste Resume Text",
    height=200,
    placeholder="Paste resume content here..."
)


# ==========================================
# JOB DESCRIPTION
# ==========================================

job_description = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the job description here..."
)


# ==========================================
# SCREEN RESUME BUTTON
# ==========================================

if st.button("🔍 Screen Resume"):

    final_resume_text = ""

    # --------------------------------------
    # Extract uploaded PDF
    # --------------------------------------

    if uploaded_file is not None:

        if uploaded_file.name.lower().endswith(".pdf"):

            final_resume_text = extract_pdf_text(
                uploaded_file
            )

        # ----------------------------------
        # Extract uploaded DOCX
        # ----------------------------------

        elif uploaded_file.name.lower().endswith(".docx"):

            final_resume_text = extract_docx_text(
                uploaded_file
            )

    # --------------------------------------
    # Use pasted resume
    # --------------------------------------

    elif resume_text.strip():

        final_resume_text = resume_text

    # --------------------------------------
    # Validate resume
    # --------------------------------------

    if not final_resume_text.strip():

        st.warning(
            " Please upload a PDF/DOCX resume "
            "or paste resume text."
        )

    # --------------------------------------
    # Validate job description
    # --------------------------------------

    elif not job_description.strip():

        st.warning(
            " Please enter a job description."
        )

    else:

        # ==================================
        # CLEAN RESUME AND JOB DESCRIPTION
        # ==================================

        cleaned_resume = clean_text(
            final_resume_text
        )

        cleaned_job = clean_text(
            job_description
        )


        # ==================================
        # RESUME CATEGORY PREDICTION
        # ==================================

        resume_vector = vectorizer.transform(
            [cleaned_resume]
        )

        prediction = model.predict(
            resume_vector
        )

        probabilities = model.predict_proba(
            resume_vector
        )

        confidence = probabilities.max() * 100


        # ==================================
        # JOB MATCH SCORE
        # ==================================

        matching_vectorizer = TfidfVectorizer()

        match_vectors = matching_vectorizer.fit_transform(
            [
                cleaned_resume,
                cleaned_job
            ]
        )

        similarity = cosine_similarity(
            match_vectors[0:1],
            match_vectors[1:2]
        )

        match_score = similarity[0][0] * 100


        # ==================================
        # SCREENING COMPLETED
        # ==================================

        st.success(
            " Resume screening completed!"
        )


        # ==================================
        # PREDICTED CATEGORY
        # ==================================

        st.subheader(
            " Predicted Job Category"
        )

        st.info(
            prediction[0]
        )


        # ==================================
        # MODEL CONFIDENCE
        # ==================================

        st.subheader(
            "🤖 Prediction Confidence"
        )

        st.progress(
            min(int(confidence), 100)
        )

        st.write(
            f"{confidence:.2f}%"
        )


        # ==================================
        # JOB MATCH SCORE
        # ==================================

        st.subheader(
            "💼 Job Match Score"
        )

        st.progress(
            min(int(match_score), 100)
        )

        st.write(
            f"{match_score:.2f}%"
        )


        # ==================================
        # FINAL RECOMMENDATION
        # ==================================

        st.subheader(
            "📋 Final Recommendation"
        )

        if match_score >= 75:

            st.success(
                "✅ SHORTLISTED"
            )

            st.write(
                "The resume has a strong similarity "
                "to the job description."
            )

        elif match_score >= 50:

            st.warning(
                "⚠️ REVIEW MANUALLY"
            )

            st.write(
                "The resume has a moderate similarity "
                "to the job description."
            )

        else:

            st.error(
                "❌ NOT RECOMMENDED"
            )

            st.write(
                "The resume has a low similarity "
                "to the job description."
            )


        # ==================================
        # EXTRACTED RESUME TEXT
        # ==================================

        st.subheader(
            "📝 Extracted Resume Text"
        )

        st.text_area(
            "Resume Content",
            final_resume_text,
            height=250
        )