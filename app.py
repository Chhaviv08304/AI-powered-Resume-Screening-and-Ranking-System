import streamlit as st
from PyPDF2 import PDFReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
 #function to extract text from pdf
def extract_text_from_pdf(file):
   pdf=PDFReader(file)
   text=""
   for page in pdf.pages:
      text += page.extract_text()
   return text

#function to rank resumes
def rank_resumes(job_description, resumes):
   #combine job descr. with resumes
   documents = [job_description] + resumes
   Vectorizer = TfidfVectorizer().fit_transform(documents)
   vectors = Vectorizer.toarray()

   #calculate cosine similarity
   job_description_vector = vectors[0]
   resume_vectors = vectors[1:]
   cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
   return cosine_similarities

#streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

#job descr. input
st.header("Job Description")
job_description = st.text_area("Enter the Job description")

#file uploader
st.header("Upload Resumes")
upload_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
 
if upload_files and job_description:
   st.header("Ranking Resumes")
 
   resumes = []
   for file in upload_files:
      text = extract_text_from_pdf(file)
      resumes.append(text)

   #Rank resumes
   score = rank_resumes(job_description, resumes)
   #display scores 
   results = pd.DataFrame({"resumes" : [file.name for file in upload_files], "score" : score })
   results = results.sort_values(by="score", ascending=False)
   
   ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)

   st.subheader("Ranked Resumes")
   for i, (file, score) in enumerate(ranked_resumes, start=1):
        st.write(f"{i}. {file.name} - Score: {score:.2f}")
