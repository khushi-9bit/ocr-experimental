import pdfplumber
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
import time

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

            #All_words.append({"text": ocr_data["text"][i], "coor_X": ocr_data["left"][i], "coor_Y": ocr_data["top"][i]})
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
    #return extracted_data

if __name__ == "__main__":
    pdf_path = r"D:\Resume_QA\QUINIQUE\Usha kumari QA Test Engineer.pdf"
    start = time.time()

    print("🔍 Extracting data from PDF...")
    output_data, final = extract_data_from_pdf(pdf_path)
    #output_data = extract_data_from_pdf(pdf_path)    
    print("Saving raw extracted data to output.json...")
    with open("outputt.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    end = time.time()
    print(f"duration: {(end - start):.4f}")
    #print(final)

# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# import json
# from collections import defaultdict, Counter
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


# def extract_words_with_layout(image):
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     words = []
#     for i in range(len(data['text'])):
#         if data['text'][i].strip():
#             word = {
#                 "text": data['text'][i],
#                 "x": data['left'][i],
#                 "y": data['top'][i],
#                 "w": data['width'][i],
#                 "h": data['height'][i],
#                 "line_y": round(data['top'][i] / 10) * 10  # Group into bands
#             }
#             words.append(word)
#     return words


# def get_frequent_bands(all_band_counts, top_n=2):
#     band_counter = Counter()
#     for band_counts in all_band_counts:
#         band_counter.update(band_counts)
#     return [band for band, _ in band_counter.most_common(top_n)]


# def extract_data_from_pdf(pdf_path):
#     extracted_data = []
#     images = convert_from_path(pdf_path, dpi=300)

#     header_bands_all = []
#     footer_bands_all = []
#     all_page_words = []

#     for image in images:
#         words = extract_words_with_layout(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
#         all_page_words.append(words)

#         y_vals = [w['y'] for w in words]
#         if not y_vals:
#             header_bands_all.append([])
#             footer_bands_all.append([])
#             continue

#         min_y, max_y = min(y_vals), max(y_vals)
#         header_bands = [w['line_y'] for w in words if w['y'] < min_y + 80]
#         footer_bands = [w['line_y'] for w in words if w['y'] > max_y - 80]

#         header_bands_all.append(header_bands)
#         footer_bands_all.append(footer_bands)

#     # Detect common header/footer bands across pages
#     top_header_bands = get_frequent_bands(header_bands_all)
#     top_footer_bands = get_frequent_bands(footer_bands_all)

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, (page, image, words) in enumerate(zip(pdf.pages, images, all_page_words)):
#             image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             borders = detect_borders_opencv(image_cv)

#             header, normal_text, bordered_text, footer = [], [], [], []

#             for word in words:
#                 word_bbox = (word['x'], word['y'], word['w'], word['h'])
#                 if word['line_y'] in top_header_bands:
#                     header.append(word['text'])
#                 elif word['line_y'] in top_footer_bands:
#                     footer.append(word['text'])
#                 elif is_text_inside_border(word_bbox, borders):
#                     bordered_text.append(word['text'])
#                 else:
#                     normal_text.append(word['text'])

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
#                 "header": " ".join(header).strip(),
#                 "normal_text": " ".join(normal_text).strip(),
#                 "bordered_text": " ".join(bordered_text).strip(),
#                 "footer": " ".join(footer).strip(),
#                 "tables": tables
#             })

#     return extracted_data


# if __name__ == "__main__":
#     pdf_path = r"D:\eco101.pdf"

#     print("🔍 Extracting data from PDF...")
#     output_data = extract_data_from_pdf(pdf_path)

#     print("Saving raw extracted data to output.json...")
#     with open("output.json", "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=4, ensure_ascii=False)
# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# import json
# from collections import defaultdict, Counter
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


# def extract_words_with_layout(image):
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     words = []
#     for i in range(len(data['text'])):
#         if data['text'][i].strip():
#             word = {
#                 "text": data['text'][i],
#                 "x": data['left'][i],
#                 "y": data['top'][i],
#                 "w": data['width'][i],
#                 "h": data['height'][i],
#                 "line_y": round(data['top'][i] / 10) * 10  # Group into bands
#             }
#             words.append(word)
#     return words


# def get_frequent_lines(lines_per_page, min_occurrence=2):
#     all_lines = Counter()
#     for lines in lines_per_page:
#         line_text = [" ".join([w['text'] for w in line]) for line in lines]
#         all_lines.update(line_text)
#     return [line for line, count in all_lines.items() if count >= min_occurrence]


# def extract_data_from_pdf(pdf_path):
#     extracted_data = []
#     images = convert_from_path(pdf_path, dpi=300)

#     header_lines_all = []
#     footer_lines_all = []
#     all_page_words = []

#     for image in images:
#         words = extract_words_with_layout(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
#         all_page_words.append(words)

#         if not words:
#             header_lines_all.append([])
#             footer_lines_all.append([])
#             continue

#         y_vals = [w['y'] for w in words]
#         heights = [w['h'] for w in words if w['h'] > 0]
#         min_y, max_y = min(y_vals), max(y_vals)
#         avg_height = np.mean(heights)

#         header_lines = defaultdict(list)
#         footer_lines = defaultdict(list)

#         for w in words:
#             if w['y'] < min_y + 100 and w['h'] >= avg_height:
#                 header_lines[w['line_y']].append(w)
#             elif w['y'] > max_y - 100 and w['h'] <= avg_height:
#                 footer_lines[w['line_y']].append(w)

#         header_lines_all.append(list(header_lines.values()))
#         footer_lines_all.append(list(footer_lines.values()))

#     repeated_headers = get_frequent_lines(header_lines_all)
#     repeated_footers = get_frequent_lines(footer_lines_all)

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, (page, image, words) in enumerate(zip(pdf.pages, images, all_page_words)):
#             image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             borders = detect_borders_opencv(image_cv)

#             header, normal_text, bordered_text, footer = [], [], [], []

#             line_map = defaultdict(list)
#             for word in words:
#                 line_map[word['line_y']].append(word)

#             for line_y, word_objs in line_map.items():
#                 line_text = " ".join([w['text'] for w in word_objs])

#                 if line_text in repeated_headers:
#                     header.extend([w['text'] for w in word_objs])
#                 elif line_text in repeated_footers:
#                     footer.extend([w['text'] for w in word_objs])
#                 else:
#                     for w in word_objs:
#                         word_bbox = (w['x'], w['y'], w['w'], w['h'])
#                         if is_text_inside_border(word_bbox, borders):
#                             bordered_text.append(w['text'])
#                         else:
#                             normal_text.append(w['text'])

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
#                 "header": " ".join(header).strip(),
#                 "normal_text": " ".join(normal_text).strip(),
#                 "bordered_text": " ".join(bordered_text).strip(),
#                 "footer": " ".join(footer).strip(),
#                 "tables": tables
#             })

#     return extracted_data


# if __name__ == "__main__":
#     pdf_path = r"D:\geo_chap_9.pdf"
#     print("🔍 Extracting data from PDF...")
#     output_data = extract_data_from_pdf(pdf_path)

#     print("Saving raw extracted data to output.json...")
#     with open("output.json", "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=4, ensure_ascii=False)
