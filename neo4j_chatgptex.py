from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from py2neo import Graph
import numpy as np

def connect_to_neo4j():
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    return graph

def filter_resumes_neo4j(graph, job_criteria):
    query = (
        "MATCH (r:Resume) WHERE "
        "r.location = $location AND "
        "r.field_experience = $field_experience AND "
        "r.experience_years >= $min_experience "
        "RETURN r.resume_id, r.skills"
    )
    return graph.run(query, job_criteria).data()

def setup_chromadb():
    client = chromadb.PersistentClient(path="/home/khushi/chroma_restored_data")
    collection = client.get_collection("default_collection", 
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2"))
    return collection

def get_top_matching_resumes(collection, job_description, top_k=5):
    job_embedding = model.encode(job_description).tolist()
    results = collection.query(query_embeddings=[job_embedding], n_results=top_k)
    return results["documents"], results["distances"]

def recommend_resumes(job_description, job_criteria):
    graph = connect_to_neo4j()
    chroma_collection = setup_chromadb()
    
    # Step 1: Filter resumes using Neo4j (structured filtering)
    filtered_resumes = filter_resumes_neo4j(graph, job_criteria)
    resume_ids = [r['r.resume_id'] for r in filtered_resumes]
    
    # Step 2: Retrieve best matches using ChromaDB (semantic search)
    matched_resumes, distances = get_top_matching_resumes(chroma_collection, job_description)
    
    # Step 3: Apply dynamic search tuning if no match is found
    if not matched_resumes:
        print("No perfect match found. Relaxing search criteria...")
        job_criteria["min_experience"] = max(0, job_criteria["min_experience"] - 1)  # Reduce experience requirement
        return recommend_resumes(job_description, job_criteria)  # Retry with relaxed criteria
    
    return matched_resumes

# Example usage
model = SentenceTransformer("all-MiniLM-L6-v2")
job_description = "Looking for a Python developer with experience in NLP and cloud computing."
job_criteria = {"location": "Remote", "field_experience": "AI", "min_experience": 3}

recommended_resumes = recommend_resumes(job_description, job_criteria)
print("Recommended Resumes:", recommended_resumes)
