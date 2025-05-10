{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8ce8fb97",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "جارٍ استخراج النص من Lippincott_Illustrated_Reviews_Pharmacology_7th.pdf ...\n",
      "حدث خطأ أثناء معالجة Lippincott_Illustrated_Reviews_Pharmacology_7th.pdf: [Errno 2] No such file or directory: 'Lippincott_Illustrated_Reviews_Pharmacology_7th.pdf'\n",
      "جارٍ استخراج النص من New-Vital-First-Aid-First-Aid-Book-112019.pdf ...\n",
      "حدث خطأ أثناء معالجة New-Vital-First-Aid-First-Aid-Book-112019.pdf: [Errno 2] No such file or directory: 'New-Vital-First-Aid-First-Aid-Book-112019.pdf'\n",
      "جارٍ استخراج النص من pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf ...\n",
      "حدث خطأ أثناء معالجة pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf: [Errno 2] No such file or directory: 'pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf'\n",
      "✅ تم استخراج جميع النصوص.\n"
     ]
    }
   ],
   "source": [
    "import PyPDF2\n",
    "\n",
    "def extract_text_from_pdf(pdf_path):\n",
    "    with open(pdf_path, 'rb') as file:\n",
    "        reader = PyPDF2.PdfReader(file)\n",
    "        full_text = ''\n",
    "        for page_num, page in enumerate(reader.pages):\n",
    "            text = page.extract_text()\n",
    "            if text:\n",
    "                full_text += f\"\\n\\n--- PAGE {page_num + 1} ---\\n\\n\"\n",
    "                full_text += text\n",
    "        return full_text\n",
    "\n",
    "books = [\n",
    "    {\"pdf\": \"Lippincott_Illustrated_Reviews_Pharmacology_7th.pdf\", \"txt\": \"data/lippincott_extracted.txt\"},\n",
    "    {\"pdf\": \"New-Vital-First-Aid-First-Aid-Book-112019.pdf\", \"txt\": \"data/first_aid_extracted.txt\"},\n",
    "    {\"pdf\": \"pain_wise_a_patients_guide_to_pain_management_1nbsped_1578264081.pdf\", \"txt\": \"data/pain_management_extracted.txt\"}\n",
    "]\n",
    "\n",
    "for book in books:\n",
    "    try:\n",
    "        print(f\"جارٍ استخراج النص من {book['pdf']} ...\")\n",
    "        pdf_text = extract_text_from_pdf(book['pdf'])\n",
    "        with open(book['txt'], 'w', encoding='utf-8') as f:\n",
    "            f.write(pdf_text)\n",
    "        print(f\"تم الانتهاء من {book['txt']}\")\n",
    "    except Exception as e:\n",
    "        print(f\"حدث خطأ أثناء معالجة {book['pdf']}: {e}\")\n",
    "\n",
    "print(\"✅ تم استخراج جميع النصوص.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2cc84fed",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ تم تقسيم كتاب الدوائية.\n"
     ]
    }
   ],
   "source": [
    "import re\n",
    "import json\n",
    "\n",
    "def split_chapters(text, keyword=\"CHAPTER\"):\n",
    "    pattern = rf'({keyword} \\d+[\\s\\S]*?)(?={keyword} \\d+|$)'\n",
    "    chapters_raw = re.findall(pattern, text, re.IGNORECASE)\n",
    "\n",
    "    chapters = []\n",
    "    for i, ch in enumerate(chapters_raw):\n",
    "        title_match = re.search(rf'{keyword} \\d+[^\\\\n]*', ch, re.IGNORECASE)\n",
    "        title = title_match.group(0) if title_match else f\"{keyword} {i + 1}\"\n",
    "        summary = ' '.join(ch.split()[:50]) + \"...\"\n",
    "\n",
    "        chapters.append({\n",
    "            \"chapter_number\": i + 1,\n",
    "            \"title\": title.strip(),\n",
    "            \"summary\": summary.strip(),\n",
    "            \"content\": ch.strip(),\n",
    "            \"keywords\": [],\n",
    "            \"image\": \"\"\n",
    "        })\n",
    "\n",
    "    return chapters\n",
    "\n",
    "with open('data/lippincott_extracted.txt', 'r', encoding='utf-8') as f:\n",
    "    full_text = f.read()\n",
    "\n",
    "chapters = split_chapters(full_text, keyword=\"CHAPTER\")\n",
    "\n",
    "with open('data/lippincott_chapters.json', 'w', encoding='utf-8') as f:\n",
    "    json.dump(chapters, f, ensure_ascii=False, indent=4)\n",
    "\n",
    "print(\"✅ تم تقسيم كتاب الدوائية.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ab0fdffc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ تم تقسيم كتاب الإسعاف الأولي.\n"
     ]
    }
   ],
   "source": [
    "import re\n",
    "import json\n",
    "\n",
    "def split_topics(text, keyword=\"Lesson\"):\n",
    "    pattern = rf'({keyword} \\d+[\\s\\S]*?)(?={keyword} \\d+|$)'\n",
    "    topics_raw = re.findall(pattern, text, re.IGNORECASE)\n",
    "\n",
    "    topics = []\n",
    "    for i, tp in enumerate(topics_raw):\n",
    "        title_match = re.search(rf'{keyword} \\d+[^\\\\n]*', tp, re.IGNORECASE)\n",
    "        title = title_match.group(0) if title_match else f\"{keyword} {i + 1}\"\n",
    "        summary = ' '.join(tp.split()[:50]) + \"...\"\n",
    "\n",
    "        topics.append({\n",
    "            \"topic_number\": i + 1,\n",
    "            \"title\": title.strip(),\n",
    "            \"summary\": summary.strip(),\n",
    "            \"content\": tp.strip(),\n",
    "            \"keywords\": [],\n",
    "            \"image\": \"\"\n",
    "        })\n",
    "\n",
    "    return topics\n",
    "\n",
    "with open('data/first_aid_extracted.txt', 'r', encoding='utf-8') as f:\n",
    "    full_text = f.read()\n",
    "\n",
    "topics = split_topics(full_text, keyword=\"Lesson\")\n",
    "\n",
    "with open('data/first_aid_topics.json', 'w', encoding='utf-8') as f:\n",
    "    json.dump(topics, f, ensure_ascii=False, indent=4)\n",
    "\n",
    "print(\"✅ تم تقسيم كتاب الإسعاف الأولي.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a0226320",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ تم تقسيم .\n"
     ]
    }
   ],
   "source": [
    "import re\n",
    "import json\n",
    "\n",
    "def split_topics(text, keyword=\"Lesson\"):\n",
    "    pattern = rf'({keyword} \\d+[\\s\\S]*?)(?={keyword} \\d+|$)'\n",
    "    topics_raw = re.findall(pattern, text, re.IGNORECASE)\n",
    "\n",
    "    topics = []\n",
    "    for i, tp in enumerate(topics_raw):\n",
    "        title_match = re.search(rf'{keyword} \\d+[^\\\\n]*', tp, re.IGNORECASE)\n",
    "        title = title_match.group(0) if title_match else f\"{keyword} {i + 1}\"\n",
    "        summary = ' '.join(tp.split()[:50]) + \"...\"\n",
    "\n",
    "        topics.append({\n",
    "            \"topic_number\": i + 1,\n",
    "            \"title\": title.strip(),\n",
    "            \"summary\": summary.strip(),\n",
    "            \"content\": tp.strip(),\n",
    "            \"keywords\": [],\n",
    "            \"image\": \"\"\n",
    "        })\n",
    "\n",
    "    return topics\n",
    "\n",
    "with open('data/pain_management_extracted.txt', 'r', encoding='utf-8') as f:\n",
    "    full_text = f.read()\n",
    "\n",
    "topics = split_topics(full_text, keyword=\"Lesson\")\n",
    "\n",
    "with open('data/pain_management_chapters.json', 'w', encoding='utf-8') as f:\n",
    "    json.dump(topics, f, ensure_ascii=False, indent=4)\n",
    "\n",
    "print(\"✅ تم تقسيم .\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "dd10c0e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ تم دمج جميع الكتب.\n"
     ]
    }
   ],
   "source": [
    "import json\n",
    "\n",
    "def load_json_file(filename):\n",
    "    with open(filename, 'r', encoding='utf-8') as f:\n",
    "        return json.load(f)\n",
    "\n",
    "lippincott_data = load_json_file('data/lippincott_chapters.json')\n",
    "first_aid_data = load_json_file('data/first_aid_topics.json')\n",
    "pain_mgmt_data = load_json_file('data/pain_management_chapters.json')\n",
    "\n",
    "for item in lippincott_data:\n",
    "    item['category'] = 'pharmacology'\n",
    "    item['language'] = 'en'\n",
    "\n",
    "for item in first_aid_data:\n",
    "    item['category'] = 'first_aid'\n",
    "    item['language'] = 'en'\n",
    "\n",
    "for item in pain_mgmt_data:\n",
    "    item['category'] = 'pain_management'\n",
    "    item['language'] = 'en'\n",
    "\n",
    "merged_data = lippincott_data + first_aid_data + pain_mgmt_data\n",
    "\n",
    "with open('data/medical_knowledge.json', 'w', encoding='utf-8') as f:\n",
    "    json.dump(merged_data, f, ensure_ascii=False, indent=4)\n",
    "\n",
    "print(\"✅ تم دمج جميع الكتب.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "94aff8e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ تم إضافة الكلمات المفتاحية.\n"
     ]
    }
   ],
   "source": [
    "import json\n",
    "import spacy\n",
    "import re\n",
    "\n",
    "nlp = spacy.load(\"en_core_web_sm\")\n",
    "\n",
    "def clean_text(text):\n",
    "    return re.sub(r'\\d+|\\W+', ' ', text.lower()).strip()\n",
    "\n",
    "def extract_keywords(text, limit=5):\n",
    "    doc = nlp(clean_text(text))\n",
    "    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and len(token.text) > 3]\n",
    "    return list(set(keywords))[:limit]\n",
    "\n",
    "with open('data/medical_knowledge.json', 'r', encoding='utf-8') as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "for item in data:\n",
    "    combined = item.get('title', '') + ' ' + item.get('summary', '') + ' ' + item.get('content', '')\n",
    "    item['keywords'] = extract_keywords(combined)\n",
    "\n",
    "with open('data/medical_knowledge_with_keywords.json', 'w', encoding='utf-8') as f:\n",
    "    json.dump(data, f, ensure_ascii=False, indent=4)\n",
    "\n",
    "print(\"✅ تم إضافة الكلمات المفتاحية.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "39383630",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import re\n",
    "from difflib import get_close_matches\n",
    "\n",
    "with open('data/medical_knowledge_with_keywords.json', 'r', encoding='utf-8') as f:\n",
    "    knowledge_base = json.load(f)\n",
    "\n",
    "def clean_text(text):\n",
    "    return re.sub(r'[^\\w\\s]', '', text.lower())\n",
    "\n",
    "def find_best_match(question):\n",
    "    question = clean_text(question)\n",
    "    matches = []\n",
    "\n",
    "    for item in knowledge_base:\n",
    "        title = clean_text(item['title'])\n",
    "        summary = clean_text(item.get('summary', ''))\n",
    "        content = clean_text(item.get('content', ''))\n",
    "\n",
    "        if question in title or question in summary or question in content:\n",
    "            matches.append(item)\n",
    "\n",
    "    if not matches:\n",
    "        all_titles = [clean_text(item['title']) for item in knowledge_base]\n",
    "        close_matches = get_close_matches(question, all_titles, n=1, cutoff=0.5)\n",
    "        if close_matches:\n",
    "            best_title = close_matches[0]\n",
    "            for item in knowledge_base:\n",
    "                if clean_text(item['title']) == best_title:\n",
    "                    return item\n",
    "\n",
    "    return matches[0] if matches else None\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c96469b7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b4fd0cd6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-10 04:24:50.698 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.910 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\acer\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-05-10 04:24:50.911 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.912 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.913 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.913 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.914 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.914 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.915 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.915 Session state does not function when running a script without `streamlit run`\n",
      "2025-05-10 04:24:50.916 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-10 04:24:50.916 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import json\n",
    "import re\n",
    "from difflib import get_close_matches\n",
    "import spacy\n",
    "import streamlit as st\n",
    "\n",
    "# تحميل النموذج اللغوي\n",
    "nlp = spacy.load(\"en_core_web_sm\")\n",
    "\n",
    "# تحميل قاعدة البيانات\n",
    "with open('data/medical_knowledge_with_keywords.json', 'r', encoding='utf-8') as f:\n",
    "    knowledge_base = json.load(f)\n",
    "\n",
    "def clean_text(text):\n",
    "    return re.sub(r'[^\\w\\s]', '', text.lower())\n",
    "\n",
    "def find_best_match(question):\n",
    "    question = clean_text(question)\n",
    "    matches = []\n",
    "\n",
    "    for item in knowledge_base:\n",
    "        title = clean_text(item['title'])\n",
    "        summary = clean_text(item.get('summary', ''))\n",
    "        content = clean_text(item.get('content', ''))\n",
    "\n",
    "        if question in title or question in summary or question in content:\n",
    "            matches.append(item)\n",
    "\n",
    "    if not matches:\n",
    "        all_titles = [clean_text(item['title']) for item in knowledge_base]\n",
    "        close_matches = get_close_matches(question, all_titles, n=1, cutoff=0.5)\n",
    "        if close_matches:\n",
    "            best_title = close_matches[0]\n",
    "            for item in knowledge_base:\n",
    "                if clean_text(item['title']) == best_title:\n",
    "                    return item\n",
    "\n",
    "    return matches[0] if matches else None\n",
    "\n",
    "# واجهة المستخدم\n",
    "st.title(\"🤖 الشات بوت الطبي\")\n",
    "st.markdown(\"اكتب سؤالك الطبي وسيقوم البوت بإيجاد المعلومات المناسبة من الكتب.\")\n",
    "\n",
    "user_input = st.text_input(\"اكتب سؤالك هنا...\")\n",
    "\n",
    "if user_input:\n",
    "    result = find_best_match(user_input)\n",
    "    if result:\n",
    "        st.subheader(\"🔍 تم العثور على نتيجة:\")\n",
    "        st.markdown(f\"**العنوان:** {result['title']}\")\n",
    "        st.markdown(f\"**الملخص:** {result.get('summary', 'لا يوجد ملخص')}\")\n",
    "        st.markdown(f\"**المصدر:** {result.get('category', '-')}\")\n",
    "    else:\n",
    "        st.warning(\"عذرًا، لا توجد معلومات متاحة لهذا السؤال.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d374f8a7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
