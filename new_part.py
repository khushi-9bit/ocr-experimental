from allennlp.predictors import Predictor
import fitz
# Load pre-trained NER model from AllenNLP
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")

def extract_entities_with_ner(text):
    """Extract entities using AllenNLP's NER model."""
    prediction = predictor.predict(sentence=text)
    entities = prediction['tags']
    words = prediction['tokens']
    
    extracted_entities = { "PERSON": [], "ORG": [], "GPE": [], "LOC": [], "MONEY": [], "DATE": [] }
    
    for word, entity in zip(words, entities):
        if entity in extracted_entities:
            extracted_entities[entity].append(word)
    
    return extracted_entities

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Example Input Resume (this can be any resume PDF file)
input_resume_pdf = r"D:\resumebyrag\resume_folder\resume.pdf"
input_resume_text = extract_text_from_pdf(input_resume_pdf)

input_resume_entities = extract_entities_with_ner(input_resume_text)

# Print extracted entities
print(input_resume_entities)
