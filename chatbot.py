import os
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import streamlit as st

# === 1. استخراج النصوص من PDF ===
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + '\n'
        return full_text

# === 2. تقسيم النص إلى وحدات صغيرة (لتسهيل البحث) ===
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
st.markdown("Ask your medical question and get a smart answer from multiple books.")

user_input = st.text_input("Enter your question...")

# === 5. معالجة السؤال ===
if user_input:
    # قائمة الكتب التي تريد قراءتها
    books = [
        "New-Vital-First-Aid-First-Aid-Book-112019.pdf",
        "pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf"
    ]

    all_chunks = []

    for book in books:
        if os.path.exists(book):
            with st.spinner(f"🔍 Extracting text from '{book}'..."):
                book_text = extract_text_from_pdf(book)
                chunks = split_into_units(book_text)
                all_chunks.extend(chunks)
        else:
            st.warning(f"⚠️ Book not found: {book}")

    if not all_chunks:
        st.error("❌ No text could be extracted from any book.")
    else:
        # 🔍 Semantic Search
        chunk_embeddings = search_model.encode(all_chunks, convert_to_tensor=True)
        question_embedding = search_model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
        best_chunk_index = scores.argmax().item()
        best_chunk = all_chunks[best_chunk_index]

        st.markdown("### 📝 Best Matching Text Found:")
        st.info(best_chunk[:500] + ("..." if len(best_chunk) > 500 else ""))

        # 💬 إجابة ذكية باستخدام Transformers
        with st.spinner("🧠 Generating answer..."):
            result = qa_pipeline(question=user_input, context=best_chunk)

        st.markdown("### ✅ Smart Answer Generated:")
        st.success(result['answer'])
