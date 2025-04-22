# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# import json
# from pdf2image import convert_from_path
# import time
# from transformers import BertTokenizer, BertForTokenClassification
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import os
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline

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
#     ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     normal_text, bordered_text = [], []
#     All_words = []

#     for i in range(len(ocr_data["text"])):
#         if ocr_data["text"][i].strip():
#             word_bbox = (
#                 ocr_data["left"][i], ocr_data["top"][i],
#                 ocr_data["width"][i], ocr_data["height"][i]
#             )
#             All_words.append({"text": ocr_data["text"][i]})
#             if is_text_inside_border(word_bbox, borders):
#                 bordered_text.append(ocr_data["text"][i])
#             else:
#                 normal_text.append(ocr_data["text"][i])
#     return " ".join(normal_text), " ".join(bordered_text), All_words

# def extract_data_from_pdf(pdf_path):
#     extracted_data = []
#     images = convert_from_path(pdf_path, dpi=300)
    
#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
#             image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             borders = detect_borders_opencv(image_cv)
#             normal_text, bordered_text, All_words = extract_text_with_ocr(image_cv, borders)

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
#                 "bordered_text": bordered_text.strip(),
#                 "tables": tables
#             })
#     return extracted_data, All_words

# def convert_np(obj):
#     if isinstance(obj, (np.float32, np.float64)):
#         return float(obj)
#     elif isinstance(obj, (np.int32, np.int64)):
#         return int(obj)
#     return str(obj)

# if __name__ == "__main__":
#     pdf_path = r"D:\Resume_QA\QUINIQUE\SUNIL SINGH _RESUME...pdf"
#     start = time.time()

#     print("🔍 Extracting data from PDF...")
#     output_data, final = extract_data_from_pdf(pdf_path)
    
#     print("💾 Saving raw extracted data to output.json...")
#     with open("output.json", "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=4, ensure_ascii=False)

#     end = time.time()
#     print(f"✅ Extraction duration: {(end - start):.2f} seconds")
#     print("Current Working Directory:", os.getcwd())
 
#     print("\n🚀 Running Named Entity Recognition (NER)...")
#     model_id = "dslim/bert-base-NER"

#     # tokenizer = AutoTokenizer.from_pretrained(model_id)
#     # model = AutoModelForTokenClassification.from_pretrained(model_id)
#     # model_id = "Davlan/bert-base-multilingual-cased-ner-hrl"

#     tokenizer = BertTokenizer.from_pretrained(model_id)
#     model = BertForTokenClassification.from_pretrained(model_id)
#     #ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
#     ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=False)

#     sentence = " ".join([word["text"] for word in final])
#     entities = ner_pipeline(sentence)

#     print("\n🧠 Named Entities Found:")
#     for ent in entities:
#         print(f"{ent['word']} ({ent['entity']})")

#     print("💾 Saving NER output to ner_output.json...")
#     with open("ner_output.json", "w", encoding="utf-8") as f:
#         json.dump(entities, f, indent=4, ensure_ascii=False, default=convert_np)


import pdfplumber 
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
import time
from transformers import BertTokenizer, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

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
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    normal_text, bordered_text = [], []
    All_words = []

    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            word_bbox = (
                ocr_data["left"][i], ocr_data["top"][i],
                ocr_data["width"][i], ocr_data["height"][i]
            )
            All_words.append({"text": ocr_data["text"][i]})
            if is_text_inside_border(word_bbox, borders):
                bordered_text.append(ocr_data["text"][i])
            else:
                normal_text.append(ocr_data["text"][i])
    return " ".join(normal_text), " ".join(bordered_text), All_words

def extract_data_from_pdf(pdf_path):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            borders = detect_borders_opencv(image_cv)
            normal_text, bordered_text, All_words = extract_text_with_ocr(image_cv, borders)

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
                "bordered_text": bordered_text.strip(),
                "tables": tables
            })
    return extracted_data, All_words

def convert_np(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)

if __name__ == "__main__":
    pdf_path = r"D:\Resume_QA\QUINIQUE\SUNIL SINGH _RESUME...pdf"
    start = time.time()

    print("🔍 Extracting data from PDF...")
    output_data, final = extract_data_from_pdf(pdf_path)
    
    print("💾 Saving raw extracted data to output.json...")
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    end = time.time()
    print(f"✅ Extraction duration: {(end - start):.2f} seconds")
    print("Current Working Directory:", os.getcwd())
 
    print("\n🚀 Running Named Entity Recognition (NER)...")
    model_id = "Davlan/bert-base-multilingual-cased-ner-hrl"

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForTokenClassification.from_pretrained(model_id)
    # model_id = "Davlan/bert-base-multilingual-cased-ner-hrl"

    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertForTokenClassification.from_pretrained(model_id)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, from_tf = True)
 
    sentence = " ".join([word["text"] for word in final])
    entities = ner_pipeline(sentence)

    print("\n🧠 Named Entities Found:")
    for ent in entities:
        print(f"{ent['word']} ({ent['entity_group']})")

    print("💾 Saving NER output to ner_output.json...")
    with open("ner_output.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=4, ensure_ascii=False, default=convert_np)