import pdfplumber
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path

def detect_borders_opencv(image):
    """Detects rectangular borders using OpenCV and returns bounding boxes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    return border_boxes  # Returns list of (x, y, w, h)

def is_text_inside_border(word_bbox, borders):
    """Check if a word's bounding box is inside any detected border."""
    x, y, w, h = word_bbox
    for bx, by, bw, bh in borders:
        if bx <= x and by <= y and (bx + bw) >= (x + w) and (by + bh) >= (y + h):
            return True
    return False

def extract_text_with_ocr(image, borders):
    """Extracts text using OCR and classifies normal vs bordered text."""
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    normal_text = []
    bordered_text = []

    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            word_bbox = (
                ocr_data["left"][i], ocr_data["top"][i],
                ocr_data["width"][i], ocr_data["height"][i]
            )
            if is_text_inside_border(word_bbox, borders):
                bordered_text.append(ocr_data["text"][i])
            else:
                normal_text.append(ocr_data["text"][i])

    return " ".join(normal_text), " ".join(bordered_text)

def extract_data_from_pdf(pdf_path):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)  # Convert PDF to images

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
            page_content = {
                "page_number": page_num + 1,
                "normal_text": "",
                "bordered_text": "",
                "tables": []
            }

            # Convert PIL image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            borders = detect_borders_opencv(image_cv)

            # Extract text using OCR
            normal_text, bordered_text = extract_text_with_ocr(image_cv, borders)
            page_content["normal_text"] = normal_text.strip()
            page_content["bordered_text"] = bordered_text.strip()

            # Extract structured tables
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

    return extracted_data

def chunk_text(text, chunk_size=100, split_by="para"):
    chunks = []

    if split_by == "length":
        # Split into fixed-size chunks
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

    elif split_by == "para":
        lines = text.split("\n")  # Split by line breaks
        paragraphs = []
        current_para = ""

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # Empty line → Possible paragraph break
                if current_para:
                    paragraphs.append(current_para.strip())
                    current_para = ""
                continue

            if line.startswith(" ") or line.startswith("\t"):
                # Indented line → New paragraph
                if current_para:
                    paragraphs.append(current_para.strip())
                    current_para = line
                else:
                    current_para = line
            else:
                # Check if the previous line was short (suggests a paragraph break)
                if i > 0 and len(lines[i - 1].strip()) < 50:
                    if current_para:
                        paragraphs.append(current_para.strip())
                        current_para = line
                    else:
                        current_para = line
                else:
                    # Otherwise, it's part of the same paragraph
                    current_para += " " + line

        if current_para:
            paragraphs.append(current_para.strip())  # Add last paragraph

        # ✅ Append paragraphs to chunks (Fix)
        chunks.extend(paragraphs)

    elif split_by == "fullstop":
        # Split by full stops
        sentences = text.split(". ")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks


# ✅ Step 1: Extract Data from PDF
pdf_path = r"C:\Users\KhushiOjha\Downloads\geo_chap_9.pdf"
output_data = extract_data_from_pdf(pdf_path) 

# ✅ Step 2: Apply Chunking to Extracted Text
for page in output_data:
    page["normal_text_chunks"] = chunk_text(page["normal_text"], chunk_size=100, split_by="fullstop")
    page["bordered_text_chunks"] = chunk_text(page["bordered_text"], chunk_size=100, split_by="fullstop")

# ✅ Step 3: Save to JSON
with open("chunked_output.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("✅ Extraction and Chunking Complete! Data saved to chunked_output.json")
