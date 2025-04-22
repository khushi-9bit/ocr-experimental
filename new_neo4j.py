import os
import fitz  # PyMuPDF for PDF text extraction
import spacy
from pathlib import Path
from py2neo import Graph  # Import py2neo for Neo4j connection
import chromadb
from sentence_transformers import SentenceTransformer  # For generating embeddings

# Initialize spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize Neo4j connection using py2neo
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Khushi@Ninebit123"))  # Replace with your Neo4j credentials

# Initialize ChromaDB client
client = chromadb.Client()

# Initialize sentence-transformers model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # You can choose another model from HuggingFace

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF file
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
        
    return text

# Function to extract resume details from the text using NER
def extract_resume_details(resume_text):
    doc = nlp(resume_text)  # Using spaCy for entity extraction

    resume_data = {
        "name": None,
        "location": None,
        "skills": [],
        "experience_years": None
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            resume_data["name"] = ent.text
        elif ent.label_ == "GPE":  # Geopolitical entity for location
            resume_data["location"] = ent.text
        elif ent.label_ == "ORG" and "AI" in ent.text:  # Field experience (example with AI)
            resume_data["field_experience"] = ent.text
        elif ent.label_ == "DATE":  # Experience in years
            try:
                resume_data["experience_years"] = int(ent.text.split()[0])
            except ValueError:
                pass

    # Extract skills (can be expanded with a broader list)
    skill_keywords = {"Python", "NLP", "Machine Learning", "Cloud Computing", "Data Science", "Deep Learning"}
    words = set(resume_text.split())
    resume_data["skills"] = list(skill_keywords.intersection(words))

    return resume_data

# Function to store resume data in Neo4j using py2neo
def store_resume_in_neo4j(resume_data):
    # Ensure field_experience is always present in resume_data (even if not found in NER)
    field_experience = resume_data.get("field_experience", None)
    
    query = """
    CREATE (r:Resume {name: $name, location: $location, experience_years: $experience_years, 
                      field_experience: $field_experience, skills: $skills})
    """
    graph.run(query, name=resume_data["name"], location=resume_data["location"], 
              experience_years=resume_data["experience_years"], 
              field_experience=field_experience, 
              skills=resume_data["skills"])

# Function to store resume embeddings in ChromaDB
def store_resume_in_chromadb(resume_data, resume_text):
    collection = client.get_or_create_collection("resumes_collection")
    
    # Generate embeddings using sentence-transformers
    embedding = embedding_model.encode(resume_text).tolist()
    
    # Generate a unique ID for the resume (e.g., using the hash of the resume text)
    resume_id = f"resume_{hash(resume_text) % 1000000}"
    
    collection.add(
        ids=[resume_id],  # Pass the generated ID here
        metadatas=[{"name": resume_data["name"], "location": resume_data["location"]}],
        documents=[resume_text],
        embeddings=[embedding]
    )

# Function to extract and process resumes from a folder
def process_resumes_from_folder(folder_path):
    resume_data_list = []
    
    for pdf_file in Path(folder_path).rglob("*.pdf"):
        print(f"Processing {pdf_file}")
        
        # Extract text from the PDF
        resume_text = extract_text_from_pdf(pdf_file)
        
        # Extract resume details
        resume_data = extract_resume_details(resume_text)
        
        # Store resume data in Neo4j
        store_resume_in_neo4j(resume_data)
        
        # Store resume embeddings in ChromaDB
        store_resume_in_chromadb(resume_data, resume_text)
        
        resume_data_list.append(resume_data)
    
    return resume_data_list

# Example usage: Process PDFs in a folder and extract details
folder_path = r"D:\resumebyrag\resume_folder"  # Replace with your folder path
resumes = process_resumes_from_folder(folder_path)

# Output the extracted resume data
for resume in resumes:
    print("***************************************************************************",resume)
