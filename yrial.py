import os
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
    "Lippincott_Illustrated_Reviews_Pharmacology_7th.pdf",
    "New-Vital-First-Aid-First-Aid-Book-112019.pdf",
    "pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf"
]

# ========== Helper Functions ==========
def extract_text_from_pdf(pdf_path):
    """Improved PDF text extraction with error handling"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ''
            for page in reader.pages:
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
@st.cache_resource
def load_models():
    """Load all required AI models with progress indicators"""
    with st.spinner("🔄 Loading search model..."):
        search_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    with st.spinner("🔄 Loading QA model..."):
        qa_pipeline = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
    
    with st.spinner("🔄 Loading translation model..."):
        translator = Translator()
    
    with st.spinner("🔄 Loading summarization model..."):
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
    
    with st.spinner("🔄 Loading medical NER model..."):
        try:
            nlp = spacy.load("en_core_med7_lg")
        except:
            nlp = spacy.load("en_core_web_sm")
            st.warning("Medical NER model not found, using standard model")
    
    return {
        "search": search_model,
        "qa": qa_pipeline,
        "translator": translator,
        "summarizer": summarizer,
        "ner": nlp
    }

# ========== Knowledge Base ==========
class MedicalKnowledgeBase:
    def __init__(self):
        self.knowledge = []
        self.chunk_embeddings = None
        self.loaded = False
    
    def build_knowledge_base(self, books):
        """Build the knowledge base from PDF files"""
        if self.loaded:
            return
            
        with st.spinner("📚 Building knowledge base..."):
            all_chunks = []
            
            for book in books:
                if os.path.exists(book):
                    text = extract_text_from_pdf(book)
                    if text:
                        cleaned = clean_text(text)
                        chunks = split_into_chunks(cleaned)
                        all_chunks.extend(chunks)
            
            if all_chunks:
                self.knowledge = all_chunks
                self.chunk_embeddings = models["search"].encode(all_chunks, convert_to_tensor=True)
                self.loaded = True
    
    def semantic_search(self, query, top_k=3):
        """Find most relevant chunks for a query"""
        if not self.loaded:
            return []
            
        query_embedding = models["search"].encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        top_indices = scores.topk(top_k).indices.tolist()
        
        return [(self.knowledge[i], scores[i].item()) for i in top_indices]

# ========== Translation ==========
def translate_text(text, dest_lang='en'):
    """Translate text to target language"""
    try:
        translation = models["translator"].translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

# ========== SMS Integration ==========
def send_sms(message, phone_number):
    """Send SMS via Twilio"""
    if not all([st.secrets.get("TWILIO_ACCOUNT_SID"), 
               st.secrets.get("TWILIO_AUTH_TOKEN"),
               st.secrets.get("TWILIO_PHONE_NUMBER")]):
        st.error("SMS credentials not configured")
        return False
    
    try:
        client = Client(st.secrets["TWILIO_ACCOUNT_SID"], 
                        st.secrets["TWILIO_AUTH_TOKEN"])
        
        message = client.messages.create(
            body=message,
            from_=st.secrets["TWILIO_PHONE_NUMBER"],
            to=phone_number
        )
        return True
    except Exception as e:
        st.error(f"SMS sending failed: {str(e)}")
        return False

# ========== UI Components ==========
def setup_sidebar():
    """Setup the sidebar controls"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=MediSmart+AI", width=150)
        st.title("Settings")
        
        # Language selection
        global target_language
        target_language = st.selectbox(
            "Response Language",
            ["English", "Arabic", "French", "Spanish", "German"],
            index=0
        )
        
        # SMS options
        st.subheader("SMS Options")
        enable_sms = st.checkbox("Enable SMS Notifications")
        sms_number = st.text_input("Phone Number (with country code)")
        
        # Advanced options
        st.subheader("Advanced")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7,
            help="Higher values mean more confident answers but fewer responses"
        )
        
        return {
            "enable_sms": enable_sms,
            "sms_number": sms_number,
            "confidence_threshold": confidence_threshold
        }

def display_answer(question, answer, context, confidence):
    """Display the answer in a professional format"""
    st.subheader("MediSmart AI Response")
    
    with st.expander("💡 Answer"):
        st.markdown(f"**{answer}**")
        st.caption(f"Confidence: {confidence:.0%}")
    
    with st.expander("📖 Supporting Context"):
        st.markdown(context)
    
    with st.expander("🧠 Analysis"):
        # Extract medical entities
        doc = models["ner"](answer)
        if doc.ents:
            st.write("**Identified Medical Terms:**")
            for ent in doc.ents:
                st.write(f"- {ent.text} ({ent.label_})")
        
        # Generate summary
        summary = models["summarizer"](context, max_length=130, min_length=30, do_sample=False)
        st.write("**Summary:**")
        st.write(summary[0]['summary_text'])

# ========== Main Application ==========
def main():
    # Load models and initialize knowledge base
    global models
    models = load_models()
    knowledge_base = MedicalKnowledgeBase()
    knowledge_base.build_knowledge_base(BOOKS)
    
    # Setup UI
    st.title("🏥 MediSmart AI - Advanced Medical Assistant")
    st.markdown("""
        Welcome to MediSmart AI, your intelligent medical assistant. 
        Ask any medical question and get accurate, evidence-based answers.
    """)
    
    settings = setup_sidebar()
    
    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Question input
    question = st.text_input("Ask your medical question:", placeholder="e.g. What are the symptoms of diabetes?")
    
    # Process question
    if question:
        with st.spinner("🔍 Searching medical knowledge..."):
            # Translate non-English questions to English for processing
            if target_language != "English":
                translated_question = translate_text(question, dest_lang='en')
            else:
                translated_question = question
            
            # Semantic search for relevant content
            results = knowledge_base.semantic_search(translated_question)
            
            if results:
                best_context, confidence = results[0]
                
                if confidence >= settings["confidence_threshold"]:
                    # Get answer from QA model
                    qa_result = models["qa"](
                        question=translated_question,
                        context=best_context,
                        max_answer_len=200
                    )
                    
                    # Translate answer back to target language if needed
                    if target_language != "English":
                        answer = translate_text(qa_result['answer'], dest_lang=target_language.lower())
                    else:
                        answer = qa_result['answer']
                    
                    # Display results
                    display_answer(question, answer, best_context, confidence)
                    
                    # Add to conversation history
                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "confidence": confidence
                    })
                    
                    # SMS notification if enabled
                    if settings["enable_sms"] and settings["sms_number"]:
                        sms_message = f"MediSmart AI answer to '{question}': {answer[:160]}..."
                        if send_sms(sms_message, settings["sms_number"]):
                            st.sidebar.success("SMS notification sent!")
                else:
                    st.warning(f"⚠️ I'm not confident enough to answer (confidence: {confidence:.0%}). Please try rephrasing your question.")
            else:
                st.error("❌ No relevant information found in the knowledge base.")
    
    # Display conversation history
    if st.session_state.history:
        st.subheader("Conversation History")
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {item['question']}"):
                st.markdown(f"**A:** {item['answer']}")
                st.caption(f"Confidence: {item['confidence']:.0%}")
                
                if st.button("Resend as SMS", key=f"sms_{i}"):
                    if settings["sms_number"]:
                        sms_message = f"MediSmart AI answer to '{item['question']}': {item['answer'][:160]}..."
                        if send_sms(sms_message, settings["sms_number"]):
                            st.success("SMS resent!")
                    else:
                        st.error("Please provide a phone number in settings")

if __name__ == "__main__":
    main()
