import spacy
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_role_from_text(text, max_length=25):
    """Extract job role from the text using both NER and pattern-based regex, with length limitation."""
    role = "Unknown"
    
    # Adjusted pattern for more specific role extraction
    role_pattern = re.compile(r"(looking for|hiring|we need|seeking|job title|position)\s*[:\-]?\s*([\w\s-]+)", re.IGNORECASE)
    role_match = role_pattern.search(text)
    if role_match:
        role = role_match.group(2).strip()
    
    # Limit the role length to max_length characters
    role = role[:max_length]
    
    # Fallback: If no role is found using regex, use spaCy's NER to extract role (as ORG or other entities)
    if role == "Unknown":
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":  # ORG label may incorrectly map roles in some cases
                role = ent.text
    
    return role

def extract_academic_qualifications(text):
    """Extract last academic qualification and stream of study using regex and NER."""
    degree_patterns = [
        r"\b(bachelor|b\.sc|b\.a|b\.tech|bachelor's|undergraduate)\b",
        r"\b(master|m\.sc|m\.a|m\.tech|master's|graduate)\b",
        r"\b(phd|doctorate|doctoral)\b"
    ]
    
    stream_patterns = [
        r"\b(computer science|cs|information technology|it|software engineering)\b",
        r"\b(electrical engineering|electronics|ee)\b",
        r"\b(mechanical engineering|mechanical)\b",
        r"\b(civil engineering|civil)\b",
        r"\b(economics|finance|business)\b",
        # Add more patterns for streams as needed
    ]
    
    academic_qualification = "Unknown"
    stream_of_study = "Unknown"
    
    for pattern in degree_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            academic_qualification = match.group(0)
            break  # Stop after first match
    
    for pattern in stream_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            stream_of_study = match.group(0)
            break  # Stop after first match
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":  # Organizations may correspond to educational institutions
            academic_qualification = f"Education from {ent.text}"
    
    return academic_qualification, stream_of_study

def extract_work_of_art(text):
    """Extract references to work of art from text using spaCy NER (WORK_OF_ART entity)."""
    doc = nlp(text)
    work_of_art = "None"
    
    for ent in doc.ents:
        if ent.label_ == "WORK_OF_ART":  # Look for the WORK_OF_ART entity label
            work_of_art = ent.text
            break  # Stop after first match
    
    return work_of_art

def extract_entities_with_ner(text):
    """Extract role, field experience, years of experience, country using NER and custom role extraction."""
    doc = nlp(text)
    
    role = extract_role_from_text(text)
    years_of_experience = "Freshers"
    country = "Any"
    field_experience = "Unknown"
    
    # Explicitly look for GPE (geopolitical entities like countries or locations)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # Geopolitical entities (countries, locations)
            if "Reactjs" not in ent.text:  # Add checks to avoid matching tech names
                country = ent.text
        elif ent.label_ == "DATE":  # Dates (could be years or experience)
            years_of_experience = ent.text
    
    field_experience = text  # As a placeholder for now
    
    # Extract work of art
    work_of_art = extract_work_of_art(text)
    
    return {
        "role": role,
        "field_experience": field_experience,
        "years_of_experience": years_of_experience,
        "country": country,
        "work_of_art": work_of_art
    }

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def rank_resumes(job_description, resume_pdfs):
    """Rank resumes based on similarity to the job description."""
    job_entities = extract_entities_with_ner(job_description)
    job_text = f"{job_entities['role']} {job_entities['field_experience']} {job_entities['years_of_experience']} {job_entities['country']} {job_entities['work_of_art']}"

    # Extract academic qualifications and business functions
    academic_qualification, stream_of_study = extract_academic_qualifications(job_description)

    
    print("Job Description Extracted Entities:")
    print(f"Role: {job_entities['role']}")
    print(f"Field Experience: {job_entities['field_experience']}")
    print(f"Years of Experience: {job_entities['years_of_experience']}")
    print(f"Location: {job_entities['country']}")
    print(f"Academic Qualification: {academic_qualification}")
    print(f"Stream of Study: {stream_of_study}")
    print(f"Work of Art: {job_entities['work_of_art']}")
    print("="*40)

    resume_entities = []
    resume_texts = []
    for pdf_path in resume_pdfs:
        resume_text = extract_text_from_pdf(pdf_path)
        resume_entities.append(extract_entities_with_ner(resume_text))
        resume_texts.append(f"{resume_entities[-1]['role']} {resume_entities[-1]['field_experience']} {resume_entities[-1]['years_of_experience']} {resume_entities[-1]['country']} {resume_entities[-1]['work_of_art']}")

    # Vectorize the job description and resumes using TF-IDF
    vectorizer = TfidfVectorizer().fit([job_text] + resume_texts)
    job_vector = vectorizer.transform([job_text])
    resume_vectors = vectorizer.transform(resume_texts)

    # Compute cosine similarity between the job description and each resume
    similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()

    # Sort resumes by similarity score (descending order)
    ranked_resumes = sorted(zip(resume_pdfs, similarity_scores), key=lambda x: x[1], reverse=True)
    
    return ranked_resumes, similarity_scores, resume_entities

# Example Job Description
job_description = """
Skills:
Java, C++, Python, Object-oriented design, Database management, Software testing, Problem-solving, Teamwork, Communication,

Company Overview

Established in 2017, Minteworld is a Bangalore-based software development company that caters to businesses across India. They offer recruitment and digital marketing services, providing a comprehensive solution for growing companies.

Job Overview

Entry level software developer role in Minteworld, a Bangalore-based software development company. Full-Time and Remote position in Hyderabad. Salary range: Competitive. Employee count: 51-200. Freshers with less than 1 year of experience can apply.

Qualifications And Skills

Proficiency in Java, C++, and Python
Knowledge of object-oriented design principles
Experience with database management
Strong software testing and problem-solving skills
Ability to work in a team environment
Excellent communication skills

Roles And Responsibilities

Develop and maintain software applications using Java, C++, and Python
Collaborate with senior developers to design and implement new features
Assist in software testing and debugging processes
Contribute to database management tasks
Participate in team meetings and collaborate on projects

"""

# List of Resume PDF file paths
resume_pdfs = [
    r"D:\resumebyrag\resume_folder\resume.pdf",
    r"D:\resumebyrag\resume_folder\Ashutosh Pandey.pdf",
    r"D:\resumebyrag\resume_folder\Aditi Jain.pdf"
]

# Rank the resumes based on similarity to the job description
ranked_resumes, similarity_scores, resume_entities = rank_resumes(job_description, resume_pdfs)
