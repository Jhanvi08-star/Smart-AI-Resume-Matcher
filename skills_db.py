"""
skills_db.py - Technical Skill Taxonomy & Aliases
"""

SKILL_ALIASES = {
    # Programming Languages
    "python": "Python", "py": "Python", "java": "Java", "c++": "C++", "cpp": "C++",
    "c#": "C#", "csharp": "C#", "c": "C", "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript", "ruby": "Ruby", "php": "PHP",
    "swift": "Swift", "kotlin": "Kotlin", "go": "Go", "golang": "Go", "rust": "Rust",
    "r": "R", "scala": "Scala", "dart": "Dart", "html": "HTML5", "html5": "HTML5",
    "css": "CSS3", "css3": "CSS3", "sql": "SQL", "bash": "Bash",
    
    # Frameworks
    "react": "React.js", "react.js": "React.js", "reactjs": "React.js",
    "angular": "Angular", "vue": "Vue.js", "vue.js": "Vue.js",
    "node": "Node.js", "node.js": "Node.js", "nodejs": "Node.js",
    "express": "Express.js", "express.js": "Express.js",
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
    "spring boot": "Spring Boot", "asp.net": "ASP.NET", ".net": ".NET",
    "rest api": "RESTful APIs", "restful api": "RESTful APIs", "graphql": "GraphQL",
    "tailwind css": "Tailwind CSS", "bootstrap": "Bootstrap",
    
    # AI & ML
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "natural language processing": "NLP", "nlp": "NLP",
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "tensorflow": "TensorFlow", "tf": "TensorFlow", "pytorch": "PyTorch",
    "keras": "Keras", "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn",
    "pandas": "Pandas", "numpy": "NumPy", "matplotlib": "Matplotlib",
    "seaborn": "Seaborn", "opencv": "OpenCV", "llm": "LLMs", "llms": "LLMs",
    "generative ai": "Generative AI", "genai": "Generative AI",
    
    # Databases & Cloud
    "mysql": "MySQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mongodb": "MongoDB", "sqlite": "SQLite", "redis": "Redis", "oracle": "Oracle DB",
    "aws": "AWS", "azure": "Microsoft Azure", "gcp": "Google Cloud Platform (GCP)",
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "git": "Git", "github": "GitHub", "ci/cd": "CI/CD", "linux": "Linux",
    
    # Tools & Methodologies
    "power bi": "Power BI", "tableau": "Tableau", "jira": "Jira", "postman": "Postman",
    "agile": "Agile Methodology", "scrum": "Scrum", "oop": "OOP (Object-Oriented Programming)",
    "data structures": "Data Structures & Algorithms", "algorithms": "Data Structures & Algorithms"
}

SKILL_TAXONOMY = {
    "Programming Languages": ["Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript", "HTML5", "CSS3", "SQL", "Bash"],
    "Web & Backend Frameworks": ["React.js", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot", "RESTful APIs"],
    "AI, Machine Learning & Data Science": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Scikit-Learn", "Pandas", "NumPy", "Generative AI"],
    "Databases & Storage": ["MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis", "Oracle DB"],
    "Cloud, DevOps & Containers": ["AWS", "Microsoft Azure", "Google Cloud Platform (GCP)", "Docker", "Kubernetes", "Git", "GitHub", "CI/CD", "Linux"],
    "Tools & Methodologies": ["Power BI", "Tableau", "Jira", "Postman", "Agile Methodology", "OOP (Object-Oriented Programming)", "Data Structures & Algorithms"]
}

SKILL_TO_CATEGORY = {}
for category, skills in SKILL_TAXONOMY.items():
    for s in skills:
        SKILL_TO_CATEGORY[s] = category

# Ye line missing thi jo error de rahi thi:
ALL_SKILLS = [skill for sublist in SKILL_TAXONOMY.values() for skill in sublist]