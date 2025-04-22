import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
import fitz  # PyMuPDF

def extract_text_from_resume(pdf_path):
    try:
        # Open the PDF file
        with fitz.open(pdf_path) as doc:
            text = ""
            # Iterate through each page of the PDF
            for page in doc:
                text += page.get_text("text") + "\n"
        
        # If no text is found, raise a warning
        if not text.strip():
            print("Warning: Extracted text is empty.")
        else:
            print("Extracted text successfully!")
        
        return text.strip()

    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return ""

# Example usage:
pdf_path = r"D:\resumebyrag\resume_folder\Aditi Jain.pdf"  # Replace with the actual resume path
resume_text = extract_text_from_resume(pdf_path)

def extract_entities_with_nltk(text):
    # Tokenize and POS tagging
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    # Apply Named Entity Recognition
    tree = ne_chunk(tagged_tokens)

    entities = {"PERSON": [], "GPE": [], "ORG": [], "DATE": []}
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity_type = subtree.label()
            entity = " ".join([word for word, tag in subtree])
            if entity_type in entities:
                entities[entity_type].append(entity)

    return entities
entities = extract_entities_with_nltk(resume_text)
print(entities)
