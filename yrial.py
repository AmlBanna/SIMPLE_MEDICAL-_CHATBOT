# حل تثبيت spacy model تلقائيًا
import subprocess
import sys
import os
import gc

def install_spacy_model():
    try:
        import spacy
        spacy.load('en_core_web_sm')
    except:
        subprocess.check_call([
            sys.executable, 
            "-m", "spacy", 
            "download", "en_core_web_sm"
        ])

install_spacy_model()
gc.collect()

import re
import json
import PyPDF2
from difflib import get_close_matches
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
from googletrans import Translator

# ========== Configuration ==========
st.set_page_config(
    page_title="MediSmart AI - Advanced Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Constants ==========
BOOKS = [
    "Lippincott_Part1.pdf",
    "Lippincott_Part2.pdf",
    "New-Vital-First-Aid-First-Aid-Book-112019.pdf",
    "pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf"
]

# ========== Helper Functions ==========
def extract_text_from_pdf(pdf_path, max_pages=100):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return '\n'.join([page.extract_text() for i, page in enumerate(reader.pages) if i < max_pages and page.extract_text()])
    except Exception as e:
        st.error(f"Error extracting {pdf_path}: {str(e)}")
        return ""

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def split_into_chunks(text, chunk_size=500):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    return chunks + [current_chunk.strip()] if current_chunk else chunks

# ========== AI Models ==========
@st.cache_resource(show_spinner=False)
def load_models():
    with st.spinner("Loading AI models..."):
        return {
            "search": SentenceTransformer('paraphrase-MiniLM-L6-v2'),
            "qa": pipeline("question-answering", model="distilbert-base-cased-distilled-squad"),
            "translator": Translator(),
            "summarizer": pipeline("summarization", model="sshleifer/distilbart-cnn-12-6"),
            "ner": spacy.load("en_core_web_sm")
        }

# ========== Knowledge Base ==========
class MedicalKnowledgeBase:
    def __init__(self):
        self.knowledge = []
        self.chunk_embeddings = None
        self.loaded = False
    
    def build_knowledge_base(self, books):
        if self.loaded: return
        
        with st.spinner("Building knowledge base..."):
            all_chunks = []
            for book in books:
                if os.path.exists(book):
                    text = extract_text_from_pdf(book)
                    if text:
                        all_chunks.extend(split_into_chunks(clean_text(text)))
                        gc.collect()
            
            if all_chunks:
                self.knowledge = all_chunks
                self.chunk_embeddings = models["search"].encode(all_chunks, convert_to_tensor=True)
                self.loaded = True
                gc.collect()

    def semantic_search(self, query, top_k=3):
        if not self.loaded: return []
        query_embedding = models["search"].encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        return [(self.knowledge[i], scores[i].item()) for i in scores.topk(top_k).indices.tolist()]

# ========== UI Components ==========
def setup_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        return {
            "target_language": st.selectbox("Language", ["English", "Arabic", "French", "Spanish"], index=0),
            "confidence_threshold": st.slider("Confidence Threshold", 0.1, 1.0, 0.7)
        }

def display_answer(question, answer, context, confidence):
    st.subheader("💡 MediSmart Response")
    st.markdown(f"**{answer}**")
    with st.expander("📖 Context"):
        st.markdown(context)
    with st.expander("🔍 Analysis"):
        st.markdown("**Medical Terms:**")
        for ent in models["ner"](answer).ents:
            st.write(f"- {ent.text} ({ent.label_})")
        st.markdown("**Summary:**")
        st.write(models["summarizer"](context, max_length=130)[0]['summary_text'])

# ========== Main Application ==========
def main():
    global models
    models = load_models()
    knowledge_base = MedicalKnowledgeBase()
    knowledge_base.build_knowledge_base(BOOKS)
    
    st.title("🏥 MediSmart AI Medical Assistant")
    st.markdown("Ask medical questions and get evidence-based answers")
    
    settings = setup_sidebar()
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if question := st.text_input("Your medical question:"):
        with st.spinner("Searching..."):
            translated_q = question if settings["target_language"] == "English" else \
                          models["translator"].translate(question, dest='en').text
            
            if results := knowledge_base.semantic_search(translated_q):
                context, confidence = results[0]
                if confidence >= settings["confidence_threshold"]:
                    answer = models["qa"](question=translated_q, context=context)['answer']
                    if settings["target_language"] != "English":
                        answer = models["translator"].translate(answer, dest=settings["target_language"].lower()).text
                    
                    display_answer(question, answer, context, confidence)
                    st.session_state.history.append({"question": question, "answer": answer})
                else:
                    st.warning(f"Low confidence answer ({confidence:.0%}), try rephrasing")
            else:
                st.error("No relevant information found")

    if st.session_state.history:
        st.subheader("📜 History")
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.divider()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
