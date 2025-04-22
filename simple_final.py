import pdfplumber
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os 

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

def extract_text_with_ocr(image, borders, line_gap_threshold=15, para_gap_multiplier=2.0):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines = []
    current_line = []
    last_top = None

    # Group words into lines
    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        if not text:
            continue
        top = ocr_data["top"][i]
        if last_top is None or abs(top - last_top) <= line_gap_threshold:
            current_line.append((ocr_data["left"][i], top, text))
        else:
            lines.append(current_line)
            current_line = [(ocr_data["left"][i], top, text)]
        last_top = top
    if current_line:
        lines.append(current_line)

    # Calculate average line gap
    line_tops = [line[0][1] for line in lines if line]
    line_gaps = [j - i for i, j in zip(line_tops[:-1], line_tops[1:])]
    avg_gap = np.mean(line_gaps) if line_gaps else line_gap_threshold
    para_gap_threshold = avg_gap * para_gap_multiplier

    # Group lines into paragraphs
    paragraphs = []
    current_paragraph = ""
    last_line_top = None

    for line in lines:
        line_text = " ".join([word[2] for word in sorted(line)])
        line_top = line[0][1]

        if last_line_top is not None and (line_top - last_line_top) > para_gap_threshold:
            if current_paragraph:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = ""
        current_paragraph += line_text + " "
        last_line_top = line_top

    if current_paragraph:
        paragraphs.append(current_paragraph.strip())

    return [{"paragraph_number": idx + 1, "text": para} for idx, para in enumerate(paragraphs)]

def process_page(page_num, page, image):
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        borders = detect_borders_opencv(image_cv)
        paragraph_data = extract_text_with_ocr(image_cv, borders)

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

        return {
            "page_number": page_num + 1,
            "paragraphs": paragraph_data,
            "tables": tables
        }
    except Exception as e:
        print(f"Error processing page {page_num + 1}: {e}")
        return {
            "page_number": page_num + 1,
            "paragraphs": [],
            "tables": [],
            "error": str(e)
        }

def extract_data_from_pdf(pdf_path, max_threads=12):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)

    with pdfplumber.open(pdf_path) as pdf:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
                futures.append(executor.submit(process_page, page_num, page, image))

            for future in as_completed(futures):
                extracted_data.append(future.result())

    # Sort to maintain page order
    return sorted(extracted_data, key=lambda x: x["page_number"])

if __name__ == "__main__":
    pdf_path = r"D:\geo_chap_9.pdf"
    start = time.time()
    print("🔍 Extracting data from PDF using multithreading...")
    output_data = extract_data_from_pdf(pdf_path, max_threads=12)  # You can increase threads here
    
    print("💾 Saving extracted data to output.json...")
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    end = time.time()
    print(f"duration: {(end - start):.4f}")
    print("✅ Done!")
