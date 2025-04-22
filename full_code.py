import spacy
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sentence_transformers import SentenceTransformer, util

# Load spaCy model and Sentence-BERT model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

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

def semantic_search(query, documents):
    """Perform semantic search using Sentence-BERT for document retrieval."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    
    # Get the number of documents to retrieve (limit to available documents)
    num_results = min(len(documents), 5)
    
    return similarities.topk(k=num_results)

def build_rag_pipeline(query, resume_pdfs):
    """Pipeline to incorporate NER into the RAG architecture."""
    # Step 1: Enhance query with NER
    enhanced_query, ner_entities = update_query_with_entities(query)
    
    # Step 2: Extract text from resume PDFs
    resume_texts = [extract_text_from_pdf(pdf_path) for pdf_path in resume_pdfs]
    
    # Step 3: Semantic Search using enhanced query
    search_results = semantic_search(enhanced_query, resume_texts)
    
    # Display NER results for transparency
    #print("Extracted Entities from Query:", ner_entities)
    print("Semantic Search Results:", search_results)
    
    # Step 4: (Optional) Generate response using the top retrieved documents
    # Use top search results as context to generate response (e.g., GPT, T5)

    return search_results, ner_entities

def update_query_with_entities(query):
    """Update the query with NER entities to enhance semantic search."""
    ner_entities = extract_entities_with_ner(query)
    enhanced_query = f"{ner_entities['role']} {ner_entities['field_experience']} {ner_entities['years_of_experience']} {ner_entities['country']} {ner_entities['work_of_art']}"
    return enhanced_query, ner_entities

# Example Job Description
job_description = """
Job Title: HR Executive
Experience: More than a year
Location: Jabalpur
Industry: HR and Recruitment

Job Summary:
An HR professional with experience in hiring, HR processes, and managing employee-related operations. Involved in different aspects of recruitment, from job postings to hiring and onboarding new employees. Also responsible for payroll handling, compliance management, and improving employee satisfaction through engagement initiatives.

Key Responsibilities:
Handling recruitment for various roles across different domains.
Managing HR documents and records related to employees.
Conducting interviews and coordinating with hiring managers.
Assisting with payroll, employee benefits, and HR policies.
Engaging with employees for performance tracking and career growth.
Using technology to streamline HR processes.
"""

# List of Resume PDF file paths
resume_pdfs = [
    r"D:\resumebyrag\resume_folder\resume.pdf",
    r"D:\resumebyrag\resume_folder\Ashutosh Pandey.pdf",
    r"D:\resumebyrag\resume_folder\Aditi Jain.pdf"
]

# Rank the resumes based on similarity to the job description
ranked_resumes, similarity_scores, resume_entities = rank_resumes(job_description, resume_pdfs)

# Build RAG pipeline for semantic search
search_results, ner_entities = build_rag_pipeline(job_description, resume_pdfs)
