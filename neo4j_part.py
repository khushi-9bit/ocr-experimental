import os
import fitz  # PyMuPDF
import requests
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from py2neo import Graph

# ========================== #
# 1. CONNECT TO NEO4J
# ========================== #
def connect_to_neo4j():
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "Khushi@Ninebit123"))
        print("Connected to Neo4j successfully!")
        return graph
    except Exception as e:
        print(f"Neo4j Connection failed: {e}")
        return None

# ========================== #
# 2. EXTRACT TEXT FROM PDF
# ========================== #
def extract_text_from_pdf(pdf_path):
    """Extracts text from PDFs using PyMuPDF (better for structured text)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"

    return text.strip()

# ========================== #
# 3. EXTRACT RESUME DETAILS USING NER
# ========================== #
# Load NER model
nlp = spacy.load("en_core_web_sm")

def extract_resume_details(resume_text):
    doc = nlp(resume_text)

    resume_data = {
        "resume_id": None,
        "name": None,
        "location": None,
        "field_experience": None,
        "experience_years": None,
        "skills": []
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            resume_data["name"] = ent.text
        elif ent.label_ == "GPE":  # Geo-Political Entity (used for locations)
            resume_data["location"] = ent.text
        elif ent.label_ == "ORG":
            resume_data["field_experience"] = ent.text
        elif ent.label_ == "DATE":
            # Try to extract years of experience
            if "years" in ent.text or "year" in ent.text:
                try:
                    resume_data["experience_years"] = int(ent.text.split()[0])
                except ValueError:
                    pass

    # Extract skills (this depends on your dataset)
    skill_keywords = {"Python", "NLP", "Machine Learning", "Cloud Computing", "Data Science", "Deep Learning"}
    words = set(resume_text.split())
    resume_data["skills"] = list(skill_keywords.intersection(words))

    # Generate a unique resume ID
    resume_data["resume_id"] = f"R{hash(resume_text) % 100000}"

    return resume_data


# ========================== #
# 4. INSERT RESUME INTO NEO4J
# ========================== #
def insert_resume_into_neo4j(graph, resume_data, pdf_url):
    query = """
    CREATE (r:Resume {
        resume_id: $resume_id, 
        name: $name, 
        location: $location, 
        field_experience: $field_experience, 
        experience_years: $experience_years, 
        skills: $skills, 
        pdf_url: $pdf_url
    })
    """
    resume_data["pdf_url"] = pdf_url
    graph.run(query, resume_data)
    print(f"Resume {resume_data['resume_id']} added to Neo4j!")


# ========================== #
# 5. SETUP CHROMADB
# ========================== #
def setup_chromadb():
    client = chromadb.PersistentClient(path="/home/khushi/chroma_restored_data")

    # Get list of collections
    existing_collections = client.list_collections()

    # Check if "default_collection" exists
    if "default_collection" not in existing_collections:
        print("Collection not found, creating new one...")
        collection = client.create_collection("default_collection")
    else:
        collection = client.get_collection(
            name="default_collection",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        )

    return collection


# ========================== #
# 6. INSERT RESUME INTO CHROMADB
# ========================== #
def insert_resume_into_chromadb(collection, resume_data, extracted_text, model):
    embedding = model.encode(extracted_text).tolist()

    collection.add(
        documents=[extracted_text],
        embeddings=[embedding],
        ids=[resume_data["resume_id"]]
    )
    print(f"Resume {resume_data['resume_id']} added to ChromaDB!")


# ========================== #
# 7. QUERY CHROMADB FOR MATCHING RESUMES
# ========================== #
def get_top_matching_resumes(collection, job_description, model, top_k=5):
    job_embedding = model.encode(job_description).tolist()
    results = collection.query(query_embeddings=[job_embedding], n_results=top_k)

    return results.get("documents", []), results.get("distances", [])


# ========================== #
# 8. FILTER RESUMES FROM NEO4J
# ========================== #
def filter_resumes_neo4j(graph, job_criteria):
    if graph is None:
        return []
    
    query = """
    MATCH (r:Resume) WHERE 
        r.location CONTAINS $location AND 
        r.field_experience CONTAINS $field_experience AND 
        r.experience_years >= $min_experience
    RETURN r.resume_id AS resume_id, r.skills AS skills
    """
    
    results = graph.run(query, job_criteria).data()
    return results


# ========================== #
# 9. RECOMMEND BEST RESUMES
# ========================== #
def recommend_resumes(job_description, job_criteria, model, attempt=0, max_attempts=2):
    if attempt >= max_attempts:
        print("No suitable resumes found after multiple attempts. Exiting search.")
        return []

    graph = connect_to_neo4j()
    chroma_collection = setup_chromadb()

    # Step 1: Filter resumes using Neo4j
    filtered_resumes = filter_resumes_neo4j(graph, job_criteria)
    if not filtered_resumes:
        print("No resumes found in Neo4j. Stopping search.")
        return []

    resume_ids = [r["resume_id"] for r in filtered_resumes]

    # Step 2: Retrieve best matches using ChromaDB
    matched_resumes, distances = get_top_matching_resumes(chroma_collection, job_description, model)

    # Step 3: If no matches, relax search criteria
    if not matched_resumes:
        print("No perfect match found. Relaxing search criteria...")
        job_criteria["min_experience"] = max(0, job_criteria["min_experience"] - 1)
        return recommend_resumes(job_description, job_criteria, model, attempt + 1)

    return matched_resumes


# ========================== #
# 10. PROCESS RESUME
# ========================== #
def process_resume(pdf_path, neo4j_graph, chroma_collection, model):
    resume_text = extract_text_from_pdf(pdf_path)
    
    if not resume_text:
        print("No text extracted. Skipping this resume.")
        return

    resume_details = extract_resume_details(resume_text)
    print(f"Extracted Resume Data*************************: {resume_details}")

    # Store in Neo4j
    insert_resume_into_neo4j(neo4j_graph, resume_details, pdf_url=pdf_path)

    # Store in ChromaDB
    insert_resume_into_chromadb(chroma_collection, resume_details, resume_text, model)

    print(f"Resume {resume_details['resume_id']} processed successfully!\n")


# ========================== #
# 11. MAIN EXECUTION
# ========================== #
if __name__ == "__main__":
    # Load SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Example folder containing resumes (adjust path)
    resume_folder = r"C:\Users\KhushiOjha\Downloads\Resume_Dev\Resume_Dev"

    # Connect to Neo4j and ChromaDB
    neo4j_graph = connect_to_neo4j()
    chroma_collection = setup_chromadb()

    # Process all resumes in folder
    for resume_file in os.listdir(resume_folder):
        if resume_file.endswith(".pdf"):
            pdf_path = os.path.join(resume_folder, resume_file)
            process_resume(pdf_path, neo4j_graph, chroma_collection, model)

    # Example job description and criteria for matching
    job_description = "Looking for a Python developer with experience in NLP and cloud computing."
    job_criteria = {"location": "Remote", "field_experience": "Human Resource", "min_experience": 3}

    # Recommend Resumes
    recommended_resumes = recommend_resumes(job_description, job_criteria, model)

    #print("Recommended Resumes:", recommended_resumes)
