import pdfplumber
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
import os

def detect_borders_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    return border_boxes

def extract_text_with_ocr(image, page_num, save_debug_image=False):
    # Improve OCR with config
    custom_config = r'--oem 3 --psm 6'
    ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    all_text = []
    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            all_text.append(ocr_data["text"][i])

    # Debug: print first 500 characters
    print(f"\n📄 Page {page_num + 1} OCR Preview:\n", " ".join(all_text)[:500])

    # Debug: save image
    if save_debug_image:
        debug_path = f"debug_page_{page_num + 1}.png"
        cv2.imwrite(debug_path, image)
        print(f"🖼 Saved debug image: {debug_path}")

    return " ".join(all_text)

def extract_data_from_pdf(pdf_path, save_debug_images=False):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Extract all text via OCR
            combined_text = extract_text_with_ocr(image_cv, page_num, save_debug_images)

            # Extract tables (if any)
            tables = []
            pdf_tables = page.extract_tables()
            if not pdf_tables:
                print(f"⚠️ No structured tables found on page {page_num + 1}")
            else:
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
                "normal_text": combined_text.strip(),
                "tables": tables
            })
    return extracted_data

if __name__ == "__main__":
    pdf_path = r"D:\Resume_QA\QUINIQUE\Vishal Rajpure_selected.pdf"
    
    print("🔍 Extracting data from PDF...")
    output_data = extract_data_from_pdf(pdf_path, save_debug_images=True)  # Set to False if you don't need debug images
    
    output_file = "output.json"
    print(f"\n💾 Saving extracted data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("✅ Extraction complete.")
