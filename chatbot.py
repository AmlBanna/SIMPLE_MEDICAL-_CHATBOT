import json
import re
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import os

# عرض المسار الحالي لمساعدتنا في التصحيح
st.write("📂 Current Working Directory:", os.getcwd())
st.write("📄 Files in current directory:", os.listdir())

# التحقق من وجود الملف
file_path = "medical_knowledge_with_keywords.json"
if not os.path.exists(file_path):
    st.error(f"❌ الملف غير موجود: {file_path}")
    st.stop()
else:
    st.success(f"✅ الملف موجود: {file_path}")

# تحميل النموذج (سيتم تنزيله تلقائيًا عند التشغيل الأول)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# تحميل قاعدة البيانات
with open(file_path, 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# تحويل العناوين والملخصات إلى تعبيرات مضمنة
passages = [item['title'] + " " + item.get('summary', '') for item in knowledge_base]
embeddings = model.encode(passages, convert_to_tensor=True)

def find_best_matches(question, top_k=3):
    question_emb = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_emb, embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        result = knowledge_base[hit['corpus_id']]
        result['score'] = hit['score']
        results.append(result)
    return results

# واجهة المستخدم
st.title("🤖 الشات بوت الطبي")
st.markdown("اكتب سؤالك الطبي وسيقوم البوت بإيجاد المعلومات المناسبة من الكتب.")

user_input = st.text_input("اكتب سؤالك هنا...")

# خيار اختيار الفئة
category_filter = st.selectbox(
    "اختر فئة للبحث داخلها (اختياري):",
    ["كل الفئات", "pharmacology", "first_aid", "pain_management"]
)

if user_input:
    results = find_best_matches(user_input, top_k=3)

    if category_filter != "كل الفئات":
        results = [r for r in results if r.get('category') == category_filter]

    if results:
        st.subheader("🔍 أفضل النتائج:")
        for i, result in enumerate(results):
            with st.expander(f"🏆 النتيجة #{i+1} - درجة التشابه: {result['score']:.2f}"):
                st.markdown(f"**العنوان:** {result['title']}")
                st.markdown(f"**الفئة:** {result.get('category', '-')}")
                st.markdown(f"**الملخص:** {result.get('summary', 'لا يوجد ملخص')}")

                # عرض النص الكامل أو جزء منه
                content = result.get('content', '')
                if len(content) > 500:
                    st.markdown(f"**جزء من النص:** {content[:500]}...")
                else:
                    st.markdown(f"**النص الكامل:** {content}")
    else:
        st.warning("عذرًا، لا توجد معلومات متاحة لهذا السؤال.")
