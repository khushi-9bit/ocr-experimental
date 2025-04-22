import spacy
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_role_from_text(text, max_length=25):
    """Extract job role from the text using both NER and pattern-based regex, with length limitation."""
    # 1. Extract Role using regex patterns like "Looking for", "Hiring", etc.
    role = "Unknown"
    role_pattern = re.compile(r"(looking for|hiring|we need|seeking)\s+([\w\s-]+)", re.IGNORECASE)
    role_match = role_pattern.search(text)
    if role_match:
        role = role_match.group(2).strip()
    
    # Limit the role length to max_length characters
    role = role[:max_length]
    
    # 2. Fallback: If no role is found using regex, use spaCy's NER to extract role (as ORG)
    if role == "Unknown":
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":  # Companies or organizations can be mapped to roles
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

def extract_entities_with_ner(text):
    """Extract all relevant entities using spaCy's NER."""
    doc = nlp(text)
    
    entities = {
        "CARDINAL": [],
        "DATE": [],
        "EVENT": [],
        "FAC": [],
        "GPE": [],
        "LANGUAGE": [],
        "LAW": [],
        "LOC": [],
        "MONEY": [],
        "NORP": [],
        "ORDINAL": [],
        "ORG": [],
        "PERCENT": [],
        "PERSON": [],
        "PRODUCT": [],
        "QUANTITY": [],
        "TIME": [],
        "WORK_OF_ART": []
    }
    
    # Extract all entities based on their labels
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities

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
    job_text = " ".join([f"{key}: {' '.join(value)}" for key, value in job_entities.items()])

    # Extract academic qualifications and business functions
    academic_qualification, stream_of_study = extract_academic_qualifications(job_description)

    print("Job Description Extracted Entities:")
    for key, value in job_entities.items():
        print(f"{key}: {', '.join(value)}")
    print(f"Academic Qualification: {academic_qualification}")
    print(f"Stream of Study: {stream_of_study}")
    print("="*40)

    resume_entities = []
    resume_texts = []
    for pdf_path in resume_pdfs:
        resume_text = extract_text_from_pdf(pdf_path)
        resume_entities.append(extract_entities_with_ner(resume_text))
        resume_texts.append(" ".join([f"{key}: {' '.join(value)}" for key, value in resume_entities[-1].items()]))

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
Looking for a Software Engineer with at least 3 years of experience in Software Development. 
The candidate should have expertise in Python, JavaScript, and cloud platforms like AWS. 
Must be located in Bihar.
"""

import spacy
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_role_from_text(text, max_length=25):
    """Extract job role from the text using both NER and pattern-based regex, with length limitation."""
    # 1. Extract Role using regex patterns like "Looking for", "Hiring", etc.
    role = "Unknown"
    role_pattern = re.compile(r"(looking for|hiring|we need|seeking)\s+([\w\s-]+)", re.IGNORECASE)
    role_match = role_pattern.search(text)
    if role_match:
        role = role_match.group(2).strip()
    
    # Limit the role length to max_length characters
    role = role[:max_length]
    
    # 2. Fallback: If no role is found using regex, use spaCy's NER to extract role (as ORG)
    if role == "Unknown":
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":  # Companies or organizations can be mapped to roles
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

def extract_entities_with_ner(text):
    """Extract all relevant entities using spaCy's NER."""
    doc = nlp(text)
    
    entities = {
        "CARDINAL": [],
        "DATE": [],
        "EVENT": [],
        "FAC": [],
        "GPE": [],
        "LANGUAGE": [],
        "LAW": [],
        "LOC": [],
        "MONEY": [],
        "NORP": [],
        "ORDINAL": [],
        "ORG": [],
        "PERCENT": [],
        "PERSON": [],
        "PRODUCT": [],
        "QUANTITY": [],
        "TIME": [],
        "WORK_OF_ART": []
    }
    
    # Extract all entities based on their labels
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities

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
    job_text = " ".join([f"{key}: {' '.join(value)}" for key, value in job_entities.items()])

    # Extract academic qualifications and business functions
    academic_qualification, stream_of_study = extract_academic_qualifications(job_description)

    print("Job Description Extracted Entities:")
    for key, value in job_entities.items():
        print(f"{key}: {', '.join(value)}")
    print(f"Academic Qualification: {academic_qualification}")
    print(f"Stream of Study: {stream_of_study}")
    print("="*40)

    resume_entities = []
    resume_texts = []
    for pdf_path in resume_pdfs:
        resume_text = extract_text_from_pdf(pdf_path)
        resume_entities.append(extract_entities_with_ner(resume_text))
        resume_texts.append(" ".join([f"{key}: {' '.join(value)}" for key, value in resume_entities[-1].items()]))

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
We are hiring for Freshers Java, .Net, Android, Web Developer
Qualification : B.Tech / B.E / M.C.A
Experience : 0 - 4Years
Skills : Fresher, Java, Android, IOS Developer Javascript, Jquery, CSS, HTML, PHP
Candidate Profile :Candidate should have:
 Strong programming skills
 Excellent problem-solving skills
 Excellent communication skills
 Good to have knowledge in Javascript/Jquery/CSS/HTML/PHP etc.
"""

# List of Resume PDF file paths
resume_pdfs = [
    r"D:\resumebyrag\resume_folder\resume.pdf",
    r"D:\resumebyrag\resume_folder\Ashutosh Pandey.pdf",
    r"D:\resumebyrag\resume_folder\Aditi Jain.pdf"
]

# Rank the resumes based on similarity to the job description
ranked_resumes, similarity_scores, resume_entities = rank_resumes(job_description, resume_pdfs)
