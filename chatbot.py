import os
import PyPDF2
import json
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import streamlit as st

# === 1. استخراج النصوص من PDF (مثال على ملف واحد فقط للتجربة) ===

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + '\n'
        return full_text

# === 2. تقسيم النص إلى فصول أو وحدات (لتسهيل البحث) ===

def split_into_units(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# === 3. تحميل النماذج ===

@st.cache_resource
def load_models():
    search_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return search_model, qa_pipeline

search_model, qa_pipeline = load_models()

# === 4. واجهة المستخدم ===

st.title("🤖 Medical Chatbot - Hybrid Mode")
st.markdown("Ask your medical question and get a smart answer.")

user_input = st.text_input("Enter your question...")

# === 5. معالجة السؤال ===

if user_input:
    # 🔍 الخطوة 1: استيراد النص من PDF (مثال على كتاب واحد)
    pdf_file = "example_book.pdf"  # استبدل باسم كتابك
    if not os.path.exists(pdf_file):
        st.error(f"❌ File not found: {pdf_file}")
    else:
        with st.spinner("🔍 Extracting text from PDF..."):
            book_text = extract_text_from_pdf(pdf_file)

        # 📖 الخطوة 2: تقسيم النص إلى وحدات صغيرة
        chunks = split_into_units(book_text)

        # 🔍 الخطوة 3: البحث مع Semantic Search
        chunk_embeddings = search_model.encode(chunks, convert_to_tensor=True)
        question_embedding = search_model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
        best_chunk_index = scores.argmax().item()
        best_chunk = chunks[best_chunk_index]

        st.markdown("### 📝 Best Matching Text Found:")
        st.info(best_chunk[:500] + ("..." if len(best_chunk) > 500 else ""))

        # 💬 الخطوة 4: الاستنتاج باستخدام Transformers
        with st.spinner("🧠 Generating answer..."):
            result = qa_pipeline(question=user_input, context=best_chunk)

        st.markdown("### ✅ Smart Answer Generated:")
        st.success(result['answer'])
