# import json
# import nltk
# from nltk.tokenize import sent_tokenize
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch

# # === Grammar Model ===
# grammar_model_name = "textattack/roberta-base-CoLA"
# grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
# grammar_model = AutoModelForSequenceClassification.from_pretrained(grammar_model_name)
# grammar_model.eval()

# # === Zero-shot Classification Model for Semantic Meaning ===
# semantic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# semantic_labels = ["Meaningful", "Nonsense", "Unrelated"]

# # === Download NLTK tokenizer ===
# nltk.download('punkt')

# # === Load JSON and extract text ===
# def load_text_from_json(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     all_text = " ".join([page["normal_text"] for page in data if "normal_text" in page and page["normal_text"]])
#     return all_text

# # === Chunk into sentence groups ===
# def chunk_by_sentences(text, max_sentences=4):
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = []
#     for sentence in sentences:
#         current_chunk.append(sentence.strip())
#         if len(current_chunk) >= max_sentences:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
#     return chunks

# # === Grammar score function ===
# def grammar_acceptability_score(text):
#     inputs = grammar_tokenizer(text, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         logits = grammar_model(**inputs).logits
#     probs = torch.softmax(logits, dim=-1)
#     return float(probs[0][1])  # Probability of being grammatically acceptable

# # === Semantic Meaning Score ===
# def semantic_meaning_score(text):
#     result = semantic_classifier(text, candidate_labels=semantic_labels)
#     return {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}

# # === Process everything ===
# def process_json_file(json_path, output_path="combined_scores.txt"):
#     text_from_json = load_text_from_json(json_path)
#     sentence_chunks = chunk_by_sentences(text_from_json, max_sentences=4)

#     with open(output_path, "w", encoding="utf-8") as out:
#         for i, chunk in enumerate(sentence_chunks):
#             grammar_score = grammar_acceptability_score(chunk)
#             semantic_score = semantic_meaning_score(chunk)
#             out.write(f"Chunk {i+1}:\n{chunk}\n")
#             out.write(f"Grammar Score: {grammar_score:.4f}\n")
#             for label, score in semantic_score.items():
#                 out.write(f"{label} Score: {score:.4f}\n")
#             out.write("\n")
#             print(f"✅ Chunk {i+1} processed")

#     print(f"\n📝 All scores saved to '{output_path}'")

# # === Run the full pipeline ===
# if __name__ == "__main__":
#     process_json_file("output.json", "combined_scores.txt")
import pdfplumber
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download('punkt')

# Zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

LABELS = [
    "academic details",
    "professional details",
    "skill details",
    "personal details",
    "other details"
]

def detect_borders_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    return border_boxes

def is_text_inside_border(word_bbox, borders):
    x, y, w, h = word_bbox
    for bx, by, bw, bh in borders:
        if bx <= x and by <= y and (bx + bw) >= (x + w) and (by + bh) >= (y + h):
            return True
    return False

def extract_text_with_ocr(image, borders):
    custom_config = r'--oem 3 --psm 12'
    ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    normal_text = []
    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            word_bbox = (
                ocr_data["left"][i], ocr_data["top"][i],
                ocr_data["width"][i], ocr_data["height"][i]
            )
            normal_text.append(ocr_data["text"][i])
    return " ".join(normal_text)

def chunk_sentences(sentences, chunk_size=2):
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

def categorize_chunks(chunks):
    categorized_output = {
        "academic_details": [],
        "professional_details": [],
        "skill_details": [],
        "personal_details": [],
        "other_details": []
    }
    for chunk in chunks:
        result = classifier(chunk, LABELS, multi_label=False)
        best_label = result["labels"][0]
        key = best_label.replace(" ", "_")
        if key in categorized_output:
            categorized_output[key].append(chunk)
        else:
            categorized_output["other_details"].append(chunk)
    return categorized_output

def extract_data_from_pdf(pdf_path):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            borders = detect_borders_opencv(image_cv)
            normal_text = extract_text_with_ocr(image_cv, borders)

            sentences = sent_tokenize(normal_text)
            paragraph_chunks = chunk_sentences(sentences, chunk_size=2)
            categorized_chunks = categorize_chunks(paragraph_chunks)

            tables = []
            pdf_tables = page.extract_tables()
            if pdf_tables:
                for table in pdf_tables:
                    if table and len(table) > 1:
                        headers = table[0]
                        structured = []
                        for row in table[1:]:
                            row_dict = {
                                headers[i].strip(): row[i].strip()
                                for i in range(len(headers)) if headers[i] and row[i]
                            }
                            structured.append(row_dict)
                        if structured:
                            tables.append(structured)

            extracted_data.append({
                "page_number": page_num + 1,
                "normal_text": normal_text.strip(),
                "text_chunks": paragraph_chunks,
                "categorized_chunks": categorized_chunks,
                "tables": tables
            })
    return extracted_data

if __name__ == "__main__":
    pdf_path = r"D:\\Resume_QA\\QUINIQUE\\Vishal Rajpure_selected.pdf"

    print("🔍 Extracting data from PDF...")
    output_data = extract_data_from_pdf(pdf_path)

    print("Saving categorized extracted data to output.json...")
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("✅ Done!")
