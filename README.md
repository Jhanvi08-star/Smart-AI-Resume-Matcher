# Smart AI Resume Matcher & Skill Gap Analyzer

[[https://img.shields.io/badge/Python-3.9%2B-blue.svg]](https://www.python.org/)
[[https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg]](https://streamlit.io/)
[[https://img.shields.io/badge/NLP-TF--IDF%20%26%20Cosine%20Similarity-green.svg]](#)

> **SRMU - BCA 3rd Year (AD5B) - Odd Sem 2026-27**  
> **Group:** D  
> **Project Title:** Smart AI Resume Matcher & Skill Gap Analyzer  
> **Domain:** Artificial Intelligence / Natural Language Processing / HR-Tech

---

## 👥 Project Team (Group D)

| Sr. No. | Student Name              | University Roll No. | Domain Email ID                             | Contact No. |
| :-----: | :------------------------ | :-----------------: | :------------------------------------------ | :---------: |
|    1    | **Jhanvi Narayan** (Lead) |  `202410101360071`  | `jhanvinarayan_202410101360071@srmu.ac.in`  | 6263752966  |
|    2    | **Harshita Verma**        |  `202410101360065`  | `harshitaverma_202410101360065@srmu.ac.in`  | 9305040560  |
|    3    | **Dhananjay Yadav**       |  `202410101360067`  | `dhananjayyadav_202410101360067@srmu.ac.in` | 9598633986  |

---

## 📌 Executive Summary & Problem Statement

Modern recruitment processes receive hundreds of resumes per job opening. Manual screening is slow and prone to bias, while candidates lack visibility into why their resumes fail applicant tracking systems.

**Smart AI Resume Matcher & Skill Gap Analyzer** bridges this gap using:

1. **Contextual Textual Similarity (TF-IDF + Cosine Similarity):** Evaluates overall terminology, project summaries, and vocabulary match.
2. **Technical Skill Coverage & Gap Analysis:** Identifies matched skills, highlights critical missing skills, and categorizes candidate strengths.

---

## 🚀 Key Features

- **Multi-Format Ingestion:** Direct parsing for PDF, DOCX, and TXT resumes.
- **Robust Skill Extraction:** 300+ technical skills categorized across 6 domains with alias normalization (e.g. `NLP` ↔ `Natural Language Processing`).
- **Granular Skill Gap Analysis:** Matched competencies, missing skills, and extra candidate strengths.
- **Batch Screening Mode:** HR recruitment leaderboard with CSV export.
- **Actionable Recommendations:** Tailored suggestions to close identified skill gaps.

---

## 💻 Installation & Setup Guide

```bash
# 1. Clone repository
git clone [https://github.com/Jhanvi08-star/Smart-AI-Resume-Matcher.git](https://github.com/Jhanvi08-star/Smart-AI-Resume-Matcher.git)
cd Smart-AI-Resume-Matcher

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch web application
streamlit run app.py
```

# Project Structure

Smart-AI-Resume-Matcher/
├── app.py # Main Streamlit web application & UI
├── matcher.py # NLP engine, TF-IDF vectorizer, and similarity logic
├── skills_db.py # Comprehensive skill taxonomy & alias mapping
├── requirements.txt # Pinned project dependencies
└── README.md # Complete academic documentation & setup guide
