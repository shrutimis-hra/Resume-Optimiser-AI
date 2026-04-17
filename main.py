import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SETTINGS & CLEANING ---
st.set_page_config(page_title="AI Resume Optimizer", layout="wide")

def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    return text

def mask_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s?]?)?\(?\d{3}\)?[-.\s?]?\d{3}[-.\s?]?\d{4}\b'
    text = re.sub(email_pattern, '[EMAIL_REDACTED]', text)
    return re.sub(phone_pattern, '[PHONE_REDACTED]', text)

# --- UI DESIGN ---
st.title("🛡️ AI Resume Optimizer & Privacy Shield")
st.markdown("### Specialized for Cybersecurity & AI/ML Candidates")

# Sidebar for Job Description
st.sidebar.header("Job Requirements")
jd_input = st.sidebar.text_area("Paste the Job Description here:", height=300)

# Main Area
uploaded_file = st.file_uploader("Upload your Dataset (CSV)", type=["csv"])

if uploaded_file and jd_input:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} resumes!")

    # Process
    if st.button("Start Optimization"):
        with st.spinner('Applying Privacy Shield and Analyzing...'):
            # 1. Privacy Masking
            df['cleaned'] = df['Resume_str'].apply(lambda x: clean_text(mask_pii(str(x))))
            
            # 2. Vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            all_texts = [clean_text(jd_input)] + df['cleaned'].tolist()
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # 3. Match
            scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            df['Score'] = (scores * 100).round(2)
            
            # Display Results
            top_df = df.sort_values(by='Score', ascending=False).head(10)
            st.balloons()
            st.write("### 🏆 Top 10 Matching Resumes (Secured)")
            st.dataframe(top_df[['ID', 'Category', 'Score']])
            
            # Simple Chart
            st.bar_chart(top_df.set_index('ID')['Score'])
else:
    st.info("Please upload the Resume CSV and paste a Job Description to begin.")
