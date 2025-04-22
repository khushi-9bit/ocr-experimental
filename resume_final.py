# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# import json
# from pdf2image import convert_from_path

# def detect_borders_opencv(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     border_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
#     return border_boxes

# def is_text_inside_border(word_bbox, borders):
#     x, y, w, h = word_bbox
#     for bx, by, bw, bh in borders:
#         if bx <= x and by <= y and (bx + bw) >= (x + w) and (by + bh) >= (y + h):
#             return True
#     return False

# def extract_text_with_ocr(image, borders):
#     custom_config = r'--oem 3 --psm 12'
#     ocr_data = pytesseract.image_to_data(image,config=custom_config, output_type=pytesseract.Output.DICT)
#     normal_text = []
#     All_words = []
#     for i in range(len(ocr_data["text"])):
#         if ocr_data["text"][i].strip():
#             word_bbox = (
#                 ocr_data["left"][i], ocr_data["top"][i],
#                 ocr_data["width"][i], ocr_data["height"][i]
#             )

#             #All_words.append({"text": ocr_data["text"][i], "coor_X": ocr_data["left"][i], "coor_Y": ocr_data["top"][i]})
#             normal_text.append(ocr_data["text"][i])
#     return " ".join(normal_text)

# def extract_data_from_pdf(pdf_path):
#     extracted_data = []
#     images = convert_from_path(pdf_path, dpi=300)

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
#             image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             borders = detect_borders_opencv(image_cv)
#             normal_text = extract_text_with_ocr(image_cv, borders)

#             tables = []
#             pdf_tables = page.extract_tables()
#             if pdf_tables:
#                 for table in pdf_tables:
#                     if table and len(table) > 1:
#                         headers = table[0]
#                         structured = []
#                         for row in table[1:]:
#                             row_dict = {
#                                 headers[i].strip(): row[i].strip()
#                                 for i in range(len(headers)) if headers[i] and row[i]
#                             }
#                             structured.append(row_dict)
#                         if structured:
#                             tables.append(structured)

#             extracted_data.append({
#                 "page_number": page_num + 1,
#                 "normal_text": normal_text.strip(),
#                 "tables": tables
#             })
#     return extracted_data# , All_words

# if __name__ == "__main__":
#     pdf_path = r"D:\Resume_QA\QUINIQUE\Vishal Rajpure_selected.pdf"
    
#     print("🔍 Extracting data from PDF...")
#     output_data = extract_data_from_pdf(pdf_path)
    
#     print("Saving raw extracted data to output.json...")
#     with open("output.json", "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=4, ensure_ascii=False)

#     #print(final)

import pdfplumber
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from collections import defaultdict
import nltk

nltk.download('punkt')

# Step 1: OCR Functions
def detect_borders_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    return border_boxes

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

# Step 2: PDF Extractor
def extract_data_from_pdf(pdf_path):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            borders = detect_borders_opencv(image_cv)
            normal_text = extract_text_with_ocr(image_cv, borders)

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
                "tables": tables
            })
    return extracted_data

# Step 3: Chunker
def chunk_sentences(sentences, chunk_size=1):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks



# Main script
if __name__ == "__main__":
    pdf_path = r"D:\Resume_QA\QUINIQUE\SUNIL SINGH _RESUME...pdf"

    print("🔍 Extracting data from PDF...")
    output_data = extract_data_from_pdf(pdf_path)

    all_text = " ".join([page["normal_text"] for page in output_data])
    sentences = sent_tokenize(all_text)
    chunks = chunk_sentences(sentences, chunk_size=1)

    print("Saving raw extracted data to output.json...")
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
