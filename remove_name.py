import os
import spacy
from pdfplumber import open as open_pdf
from fpdf import FPDF
import unicodedata

nlp = spacy.load("en_core_web_sm")

def sanitize_text(text):
    """Sanitize text to remove non-ASCII characters."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open_pdf(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def anonymize_text(text):
    """Anonymize text by replacing PERSON entities with [REDACTED]."""
    doc = nlp(text)
    anonymized_text = text
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            anonymized_text = anonymized_text.replace(ent.text, "[REDACTED]")
    return anonymized_text

def create_new_pdf(anonymized_text, output_path):
    """Create a new PDF with anonymized text."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in anonymized_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

def process_pdfs(input_folder, output_folder):
    """Process all PDFs in the input folder and save anonymized versions in the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"anonymized_{os.path.splitext(filename)[0]}.pdf")

            print(f"Processing: {filename}")
            text = extract_text_from_pdf(input_path)

            anonymized_text = anonymize_text(text)
            anonymized_text = sanitize_text(anonymized_text)
            create_new_pdf(anonymized_text, output_path)

            print(f"Anonymized file saved to: {output_path}")

input_folder = r"C:\Users\KhushiOjha\Downloads\Resume_HR\Resume_HR"  
output_folder = r"D:\resumebyrag\resume_output_folder"  
process_pdfs(input_folder, output_folder)

print("All PDFs have been processed and anonymized.")
