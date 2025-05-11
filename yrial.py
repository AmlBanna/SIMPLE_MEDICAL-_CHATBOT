# حل تثبيت spacy model تلقائيًا
import subprocess
import sys
import gc
import os

def install_spacy_model():
    try:
        import spacy
        spacy.load('en_core_web_md')
    except:
        subprocess.check_call([
            sys.executable, 
            "-m", "spacy", 
            "download", "en_core_web_md"
        ])

install_spacy_model()

# تنظيف الذاكرة قبل بدء التطبيق
gc.collect()

import re
import json
import PyPDF2
import requests
from difflib import get_close_matches
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
from googletrans import Translator
from twilio.rest import Client

# ========== Configuration ==========
st.set_page_config(
    page_title="MediSmart AI - Advanced Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Constants ==========
BOOKS = [
    "Lippincott_Part1.pdf",  # تم تقسيم الكتاب الأصلي
    "Lippincott_Part2.pdf",
    "Lippincott_Part3.pdf",
    "Lippincott_Part4.pdf",
    "New-Vital-First-Aid-First-Aid-Book-112019.pdf",
    "pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf"
]

# ========== Helper Functions ==========
def extract_text_from_pdf(pdf_path, max_pages=200):
    """Improved PDF text extraction with pagination limit"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ''
            for i, page in enumerate(reader.pages):
                if i >= max_pages:  # تحديد عدد الصفحات
                    break
                text = page.extract_text()
                if text:
                    full_text += text + '\n'
            return full_text
    except Exception as e:
        st.error(f"Error extracting {pdf_path}: {str(e)}")
        return ""

def clean_text(text):
    """Better text cleaning for medical terms"""
    text = re.sub(r'\[\d+\]', '', text)  # Remove citations
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def split_into_chunks(text, chunk_size=500):
    """Improved chunking that preserves sentence boundaries"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ========== AI Models ==========
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all required AI models with memory optimization"""
    models = {}
    
    try:
        # تحميل النماذج مع إدارة الذاكرة
        with st.spinner("🔄 Loading models (this may take a few minutes)..."):
            # نموذج أخف للبحث
            models["search"] = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            # نموذج QA مخفف
            models["qa"] = pipeline(
                "question-answering", 
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
            
            # المترجم
            models["translator"] = Translator(service_urls=['translate.googleapis.com'])
            
            # نموذج تلخيص صغير
            models["summarizer"] = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                tokenizer="sshleifer/distilbart-cnn-12-6"
            )
            
            # نموذج التعرف على الكيانات الطبية
            try:
                models["ner"] = spacy.load("en_core_med7_lg")
            except:
                models["ner"] = spacy.load("en_core_web_sm")
                st.warning("Medical NER model not found, using standard model")
                
        gc.collect()
        return models
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise e

# ========== Knowledge Base ==========
class MedicalKnowledgeBase:
    def __init__(self):
        self.knowledge = []
        self.chunk_embeddings = None
        self.loaded = False
    
    def build_knowledge_base(self, books):
        """Build the knowledge base with memory management"""
        if self.loaded:
            return
            
        with st.spinner("📚 Building knowledge base (this may take a while)..."):
            all_chunks = []
            
            for book in books:
                if os.path.exists(book):
                    # معالجة خاصة للكتب الكبيرة
                    if "Lippincott" in book:
                        text = extract_text_from_pdf(book, max_pages=100)
                    else:
                        text = extract_text_from_pdf(book)
                    
                    if text:
                        cleaned = clean_text(text)
                        chunks = split_into_chunks(cleaned)
                        all_chunks.extend(chunks)
                        gc.collect()  # تنظيف الذاكرة بعد كل كتاب
            
            if all_chunks:
                self.knowledge = all_chunks
                # تقسيم الembeddings لتفادي مشاكل الذاكرة
                batch_size = 100
                embeddings = []
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i+batch_size]
                    embeddings.append(models["search"].encode(batch, convert_to_tensor=True))
                    gc.collect()
                
                self.chunk_embeddings = torch.cat(embeddings, dim=0)
                self.loaded = True
                gc.collect()

    def semantic_search(self, query, top_k=3):
        """Find most relevant chunks with memory safety"""
        if not self.loaded:
            return []
            
        try:
            query_embedding = models["search"].encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
            top_indices = scores.topk(top_k).indices.tolist()
            
            return [(self.knowledge[i], scores[i].item()) for i in top_indices]
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

# ========== بقية الدوال كما هي (Translation, SMS Integration, UI Components) ==========
# ... [ابقى جميع الدوال الأخرى كما هي في الكود الأصلي] ...

if __name__ == "__main__":
    # تحسين إدارة الذاكرة
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    
    main()
