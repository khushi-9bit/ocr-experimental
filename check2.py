import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import os
import json

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_tables(image, original_image, output_folder, page_num):
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_h)
    
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_v)
    
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    os.makedirs(output_folder, exist_ok=True)
    
    table_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            table_crop = original_image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_folder, f'table_page_{page_num}_{table_count}.png'), table_crop)
            table_count += 1

def extract_text_and_tables(pdf_path, output_folder):
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_content = {"page_number": page_num + 1, "text": "", "tables": []}
            
            # Extract text column-wise
            text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=True)
            if text:
                page_content["text"] = text.strip()
    
            # Extract tables with structured format
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table and len(table) > 1:
                        headers = table[0]
                        structured_table = []

                        for row in table[1:]:
                            row_dict = {}
                            for i in range(len(headers)):
                                if headers[i] and row[i]:
                                    row_dict[headers[i].strip()] = row[i].strip()
                            structured_table.append(row_dict)
                        
                        if structured_table:
                            page_content["tables"].append(structured_table)
            
            extracted_data.append(page_content)
    
    images = convert_from_path(pdf_path, dpi=300)
    os.makedirs(output_folder, exist_ok=True)
    
    for i, img in enumerate(images):
        img_cv = np.array(img)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        _, img_bin = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if not page_content["text"]:
            page_content["text"] = pytesseract.image_to_string(img_bin)
        
        processed_image = preprocess_image(img_cv)
        detect_tables(processed_image, img_cv, output_folder, i)
    
    return extracted_data

if __name__ == "__main__":
    pdf_path = r"C:\Users\KhushiOjha\Downloads\geo_chap_9.pdf"  # Change this to your PDF path
    output_folder = "output_tables"
    extracted_data = extract_text_and_tables(pdf_path, output_folder)
    
    # Save extracted data to JSON
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)
    
    print("✅ Extraction Complete. Data saved to output.json")
    print("Table detection complete. Check the output_tables folder.")
    print("Data:",extracted_data)