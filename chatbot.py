import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os


st.write("📂 Current Working Directory:", os.getcwd())
st.write("📄 Files in current directory:", os.listdir())

file_path = "medical_knowledge_with_keywords.json"
if not os.path.exists(file_path):
    st.error(f"❌ File not found: {file_path}")
    st.stop()
else:
    st.success(f"✅ File is found: {file_path}")


with open(file_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

passages = [item["title"] + " " + item.get("summary", "") for item in knowledge_base]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(passages)

def find_best_match(question):
    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, tfidf_matrix).flatten()
    best_idx = scores.argmax()
    return knowledge_base[best_idx]

st.title("🤖 Simple_Medical_ChatBot")
st.markdown("ASK YOUR QUESTION AND I WILL HELP YOU TO FIND THE ANSWER")

user_input = st.text_input("Write your question here...")

if user_input:
    result = find_best_match(user_input)
    if result:
        st.subheader("🔍 result found:")
        st.markdown(f"**title:** {result['title']}")
        st.markdown(f"**summary:** {result.get('summary', 'no summary')}")
        st.markdown(f"**resource:** {result.get('category', '-')}")
    else:
        st.warning("Sorry, no information is available for this question..")
