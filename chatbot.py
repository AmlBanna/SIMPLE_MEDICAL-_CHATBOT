import json
import os
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Check current directory (for debugging)
st.write("📂 Current Working Directory:", os.getcwd())
st.write("📄 Files in current directory:", os.listdir())

# File path
file_path = "medical_knowledge_with_keywords.json"
if not os.path.exists(file_path):
    st.error(f"❌ File not found: {file_path}")
    st.stop()
else:
    st.success(f"✅ File found: {file_path}")

# Load semantic model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load knowledge base
with open(file_path, 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# Encode passages
passages = [item['title'] + " " + item.get('summary', '') for item in knowledge_base]
embeddings = model.encode(passages, convert_to_tensor=True)

# Search function
def find_best_matches(question, top_k=3):
    question_emb = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_emb, embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        result = knowledge_base[hit['corpus_id']]
        result['score'] = hit['score']
        results.append(result)
    return results

# Streamlit UI (in English)
st.title("🤖 Simple Medical Chatbot")
st.markdown("Ask your medical question and get information from books.")

user_input = st.text_input("Enter your question here...")

# Category filter
category_filter = st.selectbox(
    "Filter by category (optional):",
    ["All categories", "pharmacology", "first_aid", "pain_management"]
)

if user_input:
    results = find_best_matches(user_input, top_k=3)

    if category_filter != "All categories":
        results = [r for r in results if r.get('category') == category_filter]

    if results:
        st.subheader("🔍 Top Results:")
        for i, result in enumerate(results):
            with st.expander(f"🏆 Result #{i+1} - Similarity Score: {result['score']:.2f}"):
                st.markdown(f"**Title:** {result['title']}")
                st.markdown(f"**Category:** {result.get('category', '-')}")
                st.markdown(f"**Summary:** {result.get('summary', 'No summary available')}")

                # Display content snippet
                content = result.get('content', '')
                if len(content) > 500:
                    st.markdown(f"**Snippet from text:** {content[:500]}...")
                else:
                    st.markdown(f"**Full text:** {content}")
    else:
        st.warning("Sorry, no matching information was found for this query.")
