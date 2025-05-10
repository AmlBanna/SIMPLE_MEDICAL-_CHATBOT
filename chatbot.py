import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os

# عرض المسار الحالي لمساعدتنا في التصحيح
st.write("📂 Current Working Directory:", os.getcwd())
st.write("📄 Files in current directory:", os.listdir())

# التحقق من وجود الملف
file_path = "data/medical_knowledge_with_keywords.json"
if not os.path.exists(file_path):
    st.error(f"❌ الملف غير موجود: {file_path}")
    st.stop()
else:
    st.success(f"✅ الملف موجود: {file_path}")

# تحميل قاعدة البيانات
with open(file_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# إنشاء نموذج البحث باستخدام الكلمات
passages = [item["title"] + " " + item.get("summary", "") for item in knowledge_base]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(passages)

def find_best_match(question):
    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, tfidf_matrix).flatten()
    best_idx = scores.argmax()
    return knowledge_base[best_idx]

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
