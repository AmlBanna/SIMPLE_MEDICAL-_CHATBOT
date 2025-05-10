import json
import re
from difflib import get_close_matches
import spacy
import streamlit as st

# تحميل النموذج اللغوي
nlp = spacy.load("en_core_web_sm")

# تحميل قاعدة البيانات
with open('data/medical_knowledge_with_keywords.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def find_best_match(question):
    question = clean_text(question)
    matches = []

    for item in knowledge_base:
        title = clean_text(item['title'])
        summary = clean_text(item.get('summary', ''))
        content = clean_text(item.get('content', ''))

        if question in title or question in summary or question in content:
            matches.append(item)

    if not matches:
        all_titles = [clean_text(item['title']) for item in knowledge_base]
        close_matches = get_close_matches(question, all_titles, n=1, cutoff=0.5)
        if close_matches:
            best_title = close_matches[0]
            for item in knowledge_base:
                if clean_text(item['title']) == best_title:
                    return item

    return matches[0] if matches else None

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
