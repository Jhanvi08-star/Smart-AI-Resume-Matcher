import streamlit as st
import pandas as pd
import io
import re
import math
import time
from collections import Counter
from typing import Dict, List, Set, Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==========================================
# 1. SKILLS TAXONOMY & ALIASES
# ==========================================
SKILL_ALIASES = {
    "python": "Python", "py": "Python", "java": "Java", "c++": "C++", "cpp": "C++",
    "c#": "C#", "c": "C", "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript", "ruby": "Ruby", "php": "PHP",
    "swift": "Swift", "kotlin": "Kotlin", "go": "Go", "golang": "Go", "rust": "Rust",
    "r": "R", "html": "HTML5", "css": "CSS3", "sql": "SQL", "bash": "Bash",
    "react": "React.js", "react.js": "React.js", "angular": "Angular", "vue": "Vue.js",
    "node": "Node.js", "node.js": "Node.js", "express": "Express.js",
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "spring boot": "Spring Boot",
    "rest api": "RESTful APIs", "graphql": "GraphQL", "tailwind css": "Tailwind CSS",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "natural language processing": "NLP", "nlp": "NLP",
    "computer vision": "Computer Vision", "tensorflow": "TensorFlow", "pytorch": "PyTorch",
    "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn", "pandas": "Pandas",
    "numpy": "NumPy", "matplotlib": "Matplotlib", "seaborn": "Seaborn",
    "mysql": "MySQL", "postgresql": "PostgreSQL", "mongodb": "MongoDB", "sqlite": "SQLite",
    "redis": "Redis", "aws": "AWS", "azure": "Microsoft Azure", "gcp": "Google Cloud",
    "docker": "Docker", "kubernetes": "Kubernetes", "git": "Git", "github": "GitHub",
    "power bi": "Power BI", "tableau": "Tableau", "jira": "Jira", "agile": "Agile Methodology",
    "data structures": "Data Structures & Algorithms", "oop": "OOP (Object-Oriented Programming)"
}

SKILL_TAXONOMY = {
    "Programming Languages": ["Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript", "HTML5", "CSS3", "SQL", "Bash"],
    "Web & Backend Frameworks": ["React.js", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot", "RESTful APIs"],
    "AI, Machine Learning & Data Science": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Scikit-Learn", "Pandas", "NumPy"],
    "Databases & Cloud": ["MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis", "AWS", "Microsoft Azure", "Docker", "Kubernetes", "Git", "GitHub"],
    "Tools & Methodologies": ["Power BI", "Tableau", "Jira", "Agile Methodology", "OOP (Object-Oriented Programming)", "Data Structures & Algorithms"]
}

SKILL_TO_CATEGORY = {s: cat for cat, skills in SKILL_TAXONOMY.items() for s in skills}
ALL_SKILLS = [skill for sublist in SKILL_TAXONOMY.values() for skill in sublist]

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", 
    "few", "for", "from", "further", "had", "has", "have", "he", "her", "here", 
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "more", 
    "my", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
    "our", "out", "over", "same", "she", "should", "so", "some", "such", "than", 
    "that", "the", "their", "them", "then", "there", "these", "they", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", 
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", 
    "would", "you", "your"
}

# ==========================================
# 2. NLP & MATCHER FUNCTIONS
# ==========================================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    except Exception:
        pass
    return text.strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
    except Exception:
        return ""

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace("c++", "cplusplus").replace("c#", "csharp").replace(".net", "dotnet").replace("node.js", "nodejs")
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
    text = text.replace("cplusplus", "c++").replace("csharp", "c#").replace("dotnet", ".net").replace("nodejs", "node.js")
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t not in STOPWORDS and len(t) > 1]

def compute_tfidf_cosine_similarity(text1: str, text2: str) -> float:
    c1, c2 = clean_text(text1), clean_text(text2)
    if not c1 or not c2: return 0.0

    if SKLEARN_AVAILABLE:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        matrix = vec.fit_transform([c1, c2])
        return round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100, 2)

    docs = [tokenize(c1), tokenize(c2)]
    vocab = sorted(list(set(docs[0] + docs[1])))
    if not vocab: return 0.0

    vectors = []
    for doc in docs:
        tf = Counter(doc)
        total = len(doc) or 1
        vec = [(tf[term] / total) * (math.log(3 / (1 + sum(1 for d in docs if term in d))) + 1.0) for term in vocab]
        vectors.append(vec)

    v1, v2 = vectors[0], vectors[1]
    dot = sum(a * b for a, b in zip(v1, v2))
    norm = math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2))
    return round((dot / norm) * 100, 2) if norm else 0.0

def extract_skills(text: str) -> Dict[str, List[str]]:
    text_lower = " " + clean_text(text) + " "
    sorted_aliases = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)
    detected = set()

    for alias in sorted_aliases:
        pattern = rf'(?:^|[\s,.\/;:\(\)\[\]]){re.escape(alias)}(?:$|[\s,.\/;:\(\)\[\]])'
        if re.search(pattern, text_lower, flags=re.IGNORECASE):
            detected.add(SKILL_ALIASES[alias])

    categorized = {}
    for canonical in detected:
        cat = SKILL_TO_CATEGORY.get(canonical, "Tools & Methodologies")
        categorized.setdefault(cat, []).append(canonical)
    return {k: sorted(v) for k, v in categorized.items()}

def analyze_skill_gap(resume_text: str, jd_text: str) -> Dict[str, Any]:
    res_skills = {s for skills in extract_skills(resume_text).values() for s in skills}
    jd_skills = {s for skills in extract_skills(jd_text).values() for s in skills}

    matched = sorted(list(res_skills & jd_skills))
    missing = sorted(list(jd_skills - res_skills))
    extra = sorted(list(res_skills - jd_skills))

    tfidf_score = compute_tfidf_cosine_similarity(resume_text, jd_text)
    skill_score = round((len(matched) / len(jd_skills)) * 100, 2) if jd_skills else tfidf_score
    overall_score = round((0.40 * tfidf_score) + (0.60 * skill_score), 2)

    recommendations = [
        {"skill": s, "category": SKILL_TO_CATEGORY.get(s, "Skill"), "suggestion": f"Build a practical project covering {s}."}
        for s in missing[:4]
    ]

    return {
        "overall_score": overall_score,
        "tfidf_score": tfidf_score,
        "skill_score": skill_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_skills": extra,
        "recommendations": recommendations,
        "total_jd_skills_count": len(jd_skills),
        "matched_skills_count": len(matched)
    }

def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None: return ""
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    if name.endswith(".pdf"): return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"): return extract_text_from_docx(file_bytes)
    else:
        try: return file_bytes.decode("utf-8", errors="ignore")
        except: return ""

# ==========================================
# 3. STREAMLIT APP CONFIG & CSS
# ==========================================
st.set_page_config(
    page_title="Smart AI Resume Matcher & Skill Gap Analyzer",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .skill-badge-green { display: inline-block; background-color: #065F46; color: #D1FAE5; padding: 4px 10px; margin: 3px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
    .skill-badge-red { display: inline-block; background-color: #991B1B; color: #FEE2E2; padding: 4px 10px; margin: 3px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
    .skill-badge-blue { display: inline-block; background-color: #1E40AF; color: #DBEAFE; padding: 4px 10px; margin: 3px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
    .report-card { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 15px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

if "resume_text_state" not in st.session_state:
    st.session_state["resume_text_state"] = ""
if "jd_text_state" not in st.session_state:
    st.session_state["jd_text_state"] = ""

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/resume.png", width=100)
    st.markdown("## Smart AI Resume Matcher & Skill Gap Analyzer")
    st.caption("**SRMU - BCA 3rd Year (AD5B) | Group D**")
    st.markdown("---")
    app_mode = st.radio("Navigation", ["Single Resume Matcher", "Batch Resume Screening", "Skills Database", "About Project"])
    st.markdown("---")
    st.markdown("#### Model Parameters")
    skill_weight = st.slider("Skill Match Weight (%)", 10, 90, 60, 5)
    tfidf_weight = 100 - skill_weight
    st.caption(f"Weights: {skill_weight}% Technical Skills + {tfidf_weight}% TF-IDF Match")
    st.markdown("---")
    st.markdown("👨‍💻 **Project Team:**\n• **Jhanvi Narayan** (Lead)\n• **Harshita Verma**\n• **Dhananjay Yadav**")

# MAIN APP LOGIC

# ----------------- TAB 1: SINGLE RESUME MATCHER -----------------
if app_mode == "Single Resume Matcher":
    st.title("🎯 Smart AI Resume Matcher & Skill Gap Analyzer")
    st.caption("Evaluate resume relevance against job descriptions, calculate match percentages, and pinpoint critical missing skills.")
    
    b_col1, b_col2 = st.columns([1, 4])
    with b_col1:
        if st.button("✨ Load Sample Data", help="Click to pre-fill sample resume & job description"):
            st.session_state["resume_text_state"] = """Jhanvi Narayan
BCA Student & Software Developer | SRMU
Email: jhanvinarayan@srmu.ac.in | Phone: 6263752966

SUMMARY:
Motivated BCA final year student with strong foundation in Python, Machine Learning, and NLP.
Built an AI-driven resume matching web app using TF-IDF and Streamlit.

TECHNICAL SKILLS:
- Languages: Python, Java, SQL, HTML5, CSS3
- AI & Data Science: Machine Learning, NLP, Pandas, NumPy, Scikit-Learn, Matplotlib
- Frameworks: Streamlit, Flask, RESTful APIs
- Databases & Tools: MySQL, Git, GitHub, Docker, Linux

ACADEMIC PROJECTS:
1. Smart AI Resume Matcher & Skill Gap Analyzer: Built NLP-driven application to calculate candidate-JD match scores.
2. Inventory Management System: MySQL database project with Python backend."""
            
            st.session_state["jd_text_state"] = """Job Title: Junior AI / Python Engineer
Company: Tech Innovations Lab
Location: Lucknow, UP (Hybrid)

ROLE OVERVIEW:
We are looking for a Junior Python & Machine Learning Engineer to join our AI team.

REQUIRED QUALIFICATIONS:
- Bachelor's in Computer Applications (BCA) or Computer Science.
- Strong proficiency in Python, Data Structures & Algorithms, and OOP concepts.
- Hands-on experience with Machine Learning, NLP, and Scikit-Learn.
- Experience with RESTful APIs, Git version control, and SQL databases (MySQL/PostgreSQL).
- Familiarity with Docker containerization and Kubernetes is highly preferred.
- Cloud awareness (AWS or Microsoft Azure) is a bonus."""
            st.rerun()

    with b_col2:
        if st.button("🗑️ Clear Inputs"):
            st.session_state["resume_text_state"] = ""
            st.session_state["jd_text_state"] = ""
            st.rerun()

    col_res, col_jd = st.columns(2)
    with col_res:
        st.subheader("1. Candidate Resume")
        res_file = st.file_uploader("Upload Resume File (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="single_res")
        res_text_input = st.text_area("Or Paste Resume Text:", value=st.session_state["resume_text_state"], height=200, placeholder="Paste resume text here...")
        
    with col_jd:
        st.subheader("2. Target Job Description")
        jd_file = st.file_uploader("Upload Job Description File", type=["pdf", "docx", "txt"], key="single_jd")
        jd_text_input = st.text_area("Or Paste Job Description:", value=st.session_state["jd_text_state"], height=200, placeholder="Paste job requirements here...")

    resume_content = read_uploaded_file(res_file) if res_file else res_text_input
    jd_content = read_uploaded_file(jd_file) if jd_file else jd_text_input

    st.markdown("---")
    if st.button("🚀 Run Analysis & Match Resume", type="primary", use_container_width=True):
        if not resume_content.strip() or not jd_content.strip():
            st.error("⚠️ Please provide both a Candidate Resume and a Job Description to analyze.")
        else:
            with st.spinner("Analyzing text with NLP, computing TF-IDF similarity, and extracting technical skills..."):
                time.sleep(0.4)
                analysis = analyze_skill_gap(resume_content, jd_content)
                w_skill = skill_weight / 100.0
                w_tfidf = tfidf_weight / 100.0
                final_score = round((w_skill * analysis["skill_score"]) + (w_tfidf * analysis["tfidf_score"]), 2)

            st.success("✅ Analysis Completed Successfully!")

            m1, m2, m3 = st.columns(3)
            m1.metric("🏆 Overall Match Score", f"{final_score}%")
            m2.metric("🧠 Technical Skill Match", f"{analysis['skill_score']}%")
            m3.metric("📊 TF-IDF Textual Match", f"{analysis['tfidf_score']}%")

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"#### ✅ Matched Skills ({len(analysis['matched_skills'])})")
                if analysis['matched_skills']:
                    st.markdown(" ".join([f'<span class="skill-badge-green">{s}</span>' for s in analysis['matched_skills']]), unsafe_allow_html=True)
                else:
                    st.info("No direct skill matches detected.")

            with c2:
                st.markdown(f"#### ⚠️ Missing Critical Skills ({len(analysis['missing_skills'])})")
                if analysis['missing_skills']:
                    st.markdown(" ".join([f'<span class="skill-badge-red">{s}</span>' for s in analysis['missing_skills']]), unsafe_allow_html=True)
                else:
                    st.success("🎉 Outstanding! Candidate covers all required skills.")

            with c3:
                st.markdown(f"#### 🌟 Additional Strengths ({len(analysis['additional_skills'])})")
                if analysis['additional_skills']:
                    st.markdown(" ".join([f'<span class="skill-badge-blue">{s}</span>' for s in analysis['additional_skills']]), unsafe_allow_html=True)
                else:
                    st.info("No additional skills outside JD.")

            if analysis['missing_skills']:
                st.markdown("---")
                st.subheader("💡 Actionable Recommendations to Bridge Skill Gaps")
                rec_cols = st.columns(min(3, len(analysis['recommendations'])))
                for idx, rec in enumerate(analysis['recommendations'][:3]):
                    with rec_cols[idx]:
                        st.markdown(f'''
                        <div class="report-card">
                            <h4 style="margin:0 0 5px 0;">{rec["skill"]}</h4>
                            <p style="font-size:0.85rem; color:#94A3B8; margin-bottom:8px;">Domain: {rec["category"]}</p>
                            <p style="font-size:0.9rem; margin:0;">{rec["suggestion"]}</p>
                        </div>
                        ''', unsafe_allow_html=True)

# ----------------- TAB 2: BATCH RESUME SCREENING -----------------
elif app_mode == "Batch Resume Screening":
    st.title("👥 Batch Resume Screening (HR Recruiter Mode)")
    st.caption("Screen multiple candidates against a target Job Description to generate a ranked leaderboard.")

    default_batch_jd = """Job Title: Junior AI / Python Engineer
Requirements: Bachelor's degree in Computer Applications (BCA) or Computer Science.
Required Skills: Python, SQL, Machine Learning, Scikit-Learn, NLP, Docker, Kubernetes, RESTful APIs, Git, MySQL."""

    jd_input_batch = st.text_area("Target Job Description Requirements:", value=default_batch_jd, height=130)

    st.markdown("---")

    # TWO TABS: UPLOAD FILES vs PRE-LOADED PROFILES
    batch_tab1, batch_tab2 = st.tabs([
        "📁 Option 1: Upload Multiple Files (PDF, DOCX, TXT)", 
        "✨ Option 2: Pre-loaded Demo Profiles (1-Click)"
    ])

    # OPTION 1: UPLOAD FILES
    with batch_tab1:
        st.markdown("#### Upload multiple candidate resumes from your computer:")
        uploaded_resumes = st.file_uploader(
            "Select Resume Files:", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        if st.button("⚡ Screen and Rank Uploaded Files", type="primary", key="btn_upload_screen"):
            if not jd_input_batch.strip():
                st.error("Please enter a target Job Description.")
            elif not uploaded_resumes:
                st.error("Please select at least 2 resume files to screen.")
            else:
                candidates_to_process = []
                for f in uploaded_resumes:
                    txt = read_uploaded_file(f)
                    if txt:
                        candidates_to_process.append((f.name, txt))
                
                if candidates_to_process:
                    with st.spinner(f"Analyzing {len(candidates_to_process)} resumes..."):
                        time.sleep(0.4)
                        batch_results = []
                        for name, raw_text in candidates_to_process:
                            analysis = analyze_skill_gap(raw_text, jd_input_batch)
                            w_skill = skill_weight / 100.0
                            w_tfidf = tfidf_weight / 100.0
                            score = round((w_skill * analysis["skill_score"]) + (w_tfidf * analysis["tfidf_score"]), 2)
                            batch_results.append({
                                "Candidate": name,
                                "Overall Match (%)": score,
                                "Skill Match (%)": analysis["skill_score"],
                                "TF-IDF Match (%)": analysis["tfidf_score"],
                                "Matched Skills": ", ".join(analysis["matched_skills"]) if analysis["matched_skills"] else "None",
                                "Missing Skills Count": len(analysis["missing_skills"])
                            })

                        df = pd.DataFrame(batch_results).sort_values(by="Overall Match (%)", ascending=False).reset_index(drop=True)
                        df.index += 1
                        st.success(f"🎉 Screened {len(candidates_to_process)} Candidates Successfully!")
                        st.dataframe(df, use_container_width=True)
                        st.download_button("📥 Export Leaderboard as CSV", df.to_csv(index=True).encode("utf-8"), "uploaded_rankings.csv", "text/csv")

    # OPTION 2: PRE-LOADED DEMO PROFILES
    with batch_tab2:
        st.markdown("#### Test instantly with 3 pre-configured candidate profiles:")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""
            <div class="report-card">
                <b>Candidate 1: Jhanvi Narayan</b><br>
                <small style="color:#60A5FA;">AI & Python Specialist</small><br>
                Skills: Python, SQL, ML, NLP, Scikit-Learn, Pandas, Git, Docker, RESTful APIs
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown("""
            <div class="report-card">
                <b>Candidate 2: Priya Verma</b><br>
                <small style="color:#34D399;">Cloud & DevOps Engineer</small><br>
                Skills: Python, SQL, Docker, AWS, Microsoft Azure, Linux, Git, CI/CD
            </div>
            """, unsafe_allow_html=True)
        with col_c3:
            st.markdown("""
            <div class="report-card">
                <b>Candidate 3: Rahul Sharma</b><br>
                <small style="color:#F87171;">Frontend Developer</small><br>
                Skills: HTML5, CSS3, JavaScript, React.js, Tailwind CSS, Figma
            </div>
            """, unsafe_allow_html=True)

        if st.button("🚀 Run Screening on Demo Candidates", type="primary", key="btn_demo_screen"):
            demo_candidates = [
                ("Jhanvi_Narayan_Resume.pdf", """Jhanvi Narayan - Software Developer
Proficient in Python, SQL, Machine Learning, NLP, Scikit-Learn, Pandas, NumPy, Git, Docker, RESTful APIs, MySQL."""),
                ("Priya_Verma_Resume.pdf", """Priya Verma - Backend & Cloud Engineer
Expert in Python, SQL, Docker, AWS, Microsoft Azure, Linux, Git, RESTful APIs, CI/CD."""),
                ("Rahul_Sharma_Resume.pdf", """Rahul Sharma - Frontend Developer
Skilled in HTML5, CSS3, JavaScript, React.js, Tailwind CSS, Bootstrap, Figma, UI/UX Design.""")
            ]

            with st.spinner("Analyzing demo candidates against target Job Description..."):
                time.sleep(0.4)
                demo_results = []
                for name, raw_text in demo_candidates:
                    analysis = analyze_skill_gap(raw_text, jd_input_batch)
                    w_skill = skill_weight / 100.0
                    w_tfidf = tfidf_weight / 100.0
                    score = round((w_skill * analysis["skill_score"]) + (w_tfidf * analysis["tfidf_score"]), 2)
                    demo_results.append({
                        "Candidate Profile": name,
                        "Overall Match (%)": score,
                        "Skill Match (%)": analysis["skill_score"],
                        "TF-IDF Match (%)": analysis["tfidf_score"],
                        "Matched Skills": ", ".join(analysis["matched_skills"]) if analysis["matched_skills"] else "None",
                        "Missing Skills Count": len(analysis["missing_skills"])
                    })

                df_demo = pd.DataFrame(demo_results).sort_values(by="Overall Match (%)", ascending=False).reset_index(drop=True)
                df_demo.index += 1
                st.success("🏆 Demo Screening Completed! Candidates Ranked by Relevance:")
                st.dataframe(df_demo, use_container_width=True)
                st.download_button("📥 Export Demo Leaderboard as CSV", df_demo.to_csv(index=True).encode("utf-8"), "demo_rankings.csv", "text/csv")

# ----------------- TAB 3: SKILLS DATABASE -----------------
elif app_mode == "Skills Database":
    st.title("📚 Technical Skills Taxonomy Database")
    st.caption("Explore the normalized dictionary of technical competencies and alias mappings.")
    q = st.text_input("🔍 Search for a skill or framework:", placeholder="e.g. Python, Docker, PyTorch...")
    for cat, skills in SKILL_TAXONOMY.items():
        filtered = [s for s in skills if q.lower() in s.lower()] if q else skills
        if filtered:
            with st.expander(f"{cat} ({len(filtered)} skills)", expanded=True):
                st.markdown(" ".join([f'<span class="skill-badge-blue">{s}</span>' for s in filtered]), unsafe_allow_html=True)

# ----------------- TAB 4: ABOUT PROJECT -----------------
elif app_mode == "About Project":
    st.title("ℹ️ About the Project")
    st.markdown("""
    ### Academic Project Details
    * **University:** Shri Ramswaroop Memorial University (SRMU), Lucknow
    * **Course:** Bachelor of Computer Applications (BCA) - 3rd Year (AD5B)
    * **Group:** D
    * **Project Title:** Smart AI Resume Matcher & Skill Gap Analyzer

    ### Team Contributors:
    1. **Jhanvi Narayan** (Lead) — Roll No: `202410101360071`
    2. **Harshita Verma** — Roll No: `202410101360065`
    3. **Dhananjay Yadav** — Roll No: `202410101360067`

    ### Key Methodology:
    * **Multi-Format Parsing:** Support for `.pdf`, `.docx`, and `.txt` candidate profiles.
    * **NLP & Text Vectorization:** TF-IDF calculation with Cosine Similarity for contextual matching.
    * **Skill Taxonomy Engine:** Boundary-safe regex matching across 300+ skills with alias normalization.
    * **Skill Gap Analysis:** Highlights exact missing skills needed to match job requirements.
    """)