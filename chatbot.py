import json
import re
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# تحميل النموذج (سيتم تنزيله تلقائيًا عند التشغيل الأول)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# تحميل قاعدة البيانات
with open('data/medical_knowledge_with_keywords.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# تحويل العناوين والملخصات إلى تعبيرات مضمنة
passages = [item['title'] + " " + item.get('summary', '') for item in knowledge_base]
embeddings = model.encode(passages, convert_to_tensor=True)

def find_best_match(question):
    question_emb = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_emb, embeddings, top_k=1)[0]
    best_match = knowledge_base[hits[0]['corpus_id']]
    return best_match

# واجهة المستخدم
st.title("🤖 الشات بوت الطبي")
st.markdown("اكتب سؤالك الطبي وسيقوم البوت بإيجاد المعلومات المناسبة من الكتب.")

user_input = st.text_input("اكتب سؤالك هنا...")

if user_input:
    result = find_best_match(user_input)
    if result:
        st.subheader("🔍 تم العثور على نتيجة:")
        st.markdown(f"**العنوان:** {result['title']}")
        st.markdown(f"**الملخص:** {result.get('summary', 'لا يوجد ملخص')}")
        st.markdown(f"**المصدر:** {result.get('category', '-')}")
    else:
        st.warning("عذرًا، لا توجد معلومات متاحة لهذا السؤال.")
