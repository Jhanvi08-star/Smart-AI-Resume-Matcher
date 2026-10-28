import re
import math
from collections import Counter
from typing import Dict, List, Set, Any
import io

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from skills_db import SKILL_ALIASES, SKILL_TAXONOMY, SKILL_TO_CATEGORY

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
        {"skill": s, "category": SKILL_TO_CATEGORY.get(s, "Skill"), "suggestion": f"Build a project covering {s}."}
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