import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path, min_text_length=1000):
    """
    Extract text from a PDF file and validate its content.
    Skip the PDF if it contains insufficient text or is image-based.
    
    Args:
        pdf_path (str): Path to the PDF file.
        min_text_length (int): Minimum text length to consider the PDF valid.

    Returns:
        str: Extracted text if valid, otherwise None.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle cases where text extraction fails
        
        # Validate text content
        if len(text.strip()) < min_text_length:
            print(f"Skipping PDF '{pdf_path}' due to insufficient text content.")
            return None
        
        print(f"Extracted text from '{pdf_path}' (length: {len(text.strip())} characters).")
        print("______________________________________________________________________")
        return text
    except Exception as e:
        print(f"Error processing PDF '{pdf_path}': {e}")
        return None


def find_best_resumes(job_description, pdf_folder, model_name='all-MiniLM-L6-v2', top_n=5, min_text_length=100):
    """
    Process resumes and find the best matches for the job description.

    Args:
        job_description (str): The job description to match resumes against.
        pdf_folder (str): Folder containing PDF resumes.
        model_name (str): Name of the pre-trained SentenceTransformer model.
        top_n (int): Number of top matches to display.
        min_text_length (int): Minimum text length for a PDF to be processed.

    Returns:
        list: Ranked resumes with similarity scores.
    """
    # Load pre-trained embedding model
    model = SentenceTransformer(model_name)

    # Embed the job description
    job_embedding = model.encode(job_description)

    # Read and encode resumes
    resumes = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Processing: {pdf_file}")
        
        # Extract and validate text from PDF
        resume_text = extract_text_from_pdf(pdf_path, min_text_length=min_text_length)
        if resume_text:
            resumes.append((pdf_file, resume_text))
        else:
            print(f"Skipped: {pdf_file} (Insufficient content)")

    # Check if any valid resumes are left
    if not resumes:
        print("No valid resumes found.")
        return []

    # Embed the resume texts
    resume_embeddings = [(file, model.encode(text)) for file, text in resumes]

    # Calculate similarity scores
    similarity_results = [
        (file, 1 - cosine(job_embedding, embedding))
        for file, embedding in resume_embeddings
    ]

    # Sort resumes by similarity score
    ranked_resumes = sorted(similarity_results, key=lambda x: x[1], reverse=True)

    # Display the top N matches
    print("\nTop Matches:")
    for i, (file, score) in enumerate(ranked_resumes[:top_n], start=1):
        print(f"Rank {i}: {file}")
        print(f"Score: {score:.4f}")
        print("-" * 50)

    return ranked_resumes[:top_n]


# Job description
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

resume_folder = r"D:\resumebyrag\resume_folder"  # Replace with your folder path containing PDF resumes

# Find the best resumes
best_resumes = find_best_resumes(job_description, resume_folder, min_text_length=100)

