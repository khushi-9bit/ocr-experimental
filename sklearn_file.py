# import cv2
# import numpy as np
# import pytesseract
# import json
# from pdf2image import convert_from_path
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# # === Step 1: OCR with Coordinates ===
# def extract_words_with_coordinates(image):
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     words = []
#     for i in range(len(data["text"])):
#         text = data["text"][i].strip()
#         if text:
#             words.append({
#                 "text": text,
#                 "coor_X": data["left"][i],
#                 "coor_Y": data["top"][i]
#             })
#     return words

# # === Step 2: Cluster Words by Y-Coordinate (lines) ===
# def cluster_by_lines(words):
#     coords = [[0, w['coor_Y']] for w in words]
#     scaled = StandardScaler().fit_transform(coords)
#     db = DBSCAN(eps=0.5, min_samples=2).fit(scaled)

#     lines = {}
#     for idx, label in enumerate(db.labels_):
#         if label == -1:
#             continue  # Skip noise
#         lines.setdefault(label, []).append(words[idx])
#     return lines

# # === Step 3: Build Text Lines from Clusters ===
# def assemble_lines(lines_dict):
#     line_texts = []
#     for label, words in sorted(lines_dict.items()):
#         sorted_words = sorted(words, key=lambda w: w['coor_X'])
#         line = " ".join(word['text'] for word in sorted_words)
#         line_texts.append(line)
#     return line_texts

# # === Step 4: Label Lines into Resume Sections ===
# def label_lines(line_texts):
#     section_keywords = {
#         "Education": ["education", "bachelor", "university", "degree"],
#         "Experience": ["experience", "developer", "engineer", "company", "role"],
#         "Skills": ["skills", "python", "selenium", "sql", "tools", "testing"],
#         "Certifications": ["certification", "certified", "jira"],
#         "Languages": ["english", "hindi", "language", "marathi"],
#         "Projects": ["project", "domain", "implemented"],
#         "Personal Info": ["email", "linkedin", "pune", "contact", "phone", "mobile"]
#     }

#     labeled = {key: [] for key in section_keywords}
#     labeled["Other"] = []

#     for line in line_texts:
#         lower = line.lower()
#         matched = False
#         for section, keywords in section_keywords.items():
#             if any(keyword in lower for keyword in keywords):
#                 labeled[section].append(line)
#                 matched = True
#                 break
#         if not matched:
#             labeled["Other"].append(line)
#     return labeled

# # === Step 5: Process the PDF ===
# def process_resume(pdf_path):
#     images = convert_from_path(pdf_path, dpi=300)
#     all_words = []
#     for image in images:
#         image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         words = extract_words_with_coordinates(image_cv)
#         all_words.extend(words)

#     clustered_lines = cluster_by_lines(all_words)
#     line_texts = assemble_lines(clustered_lines)
#     labeled_output = label_lines(line_texts)

#     print(json.dumps(labeled_output, indent=4, ensure_ascii=False))

# # === Run the full pipeline ===
# if __name__ == "__main__":
#     pdf_path = r"D:\Resume_QA\QUINIQUE\Vishal Rajpure_selected.pdf"  # Replace with your local path if needed
#     process_resume(pdf_path)
import cv2
import numpy as np
import pytesseract
import json
from pdf2image import convert_from_path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === OCR: Extract words with coordinates and bounding boxes ===
def extract_words_with_coordinates(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            rounded_Y = round(data["top"][i] / 10) * 10  # Normalize Y to group better
            words.append({
                "text": text,
                "coor_X": data["left"][i],
                "coor_Y": rounded_Y,
                "width": data["width"][i],
                "height": data["height"][i]
            })
    return words

# === Cluster words into 2D spatial blocks (invisible rectangles) ===
def cluster_words_spatially(words, n_clusters=8):
    coords = [[w['coor_X'], w['coor_Y']] for w in words]
    scaled = StandardScaler().fit_transform(coords)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(scaled)

    clusters = {}
    for idx, label in enumerate(kmeans.labels_):
        clusters.setdefault(label, []).append(words[idx])
    return clusters

# === Sort and reconstruct text from each cluster ===
def assemble_blocks_from_clusters(clusters):
    block_texts = {}
    for label, words in clusters.items():
        sorted_words = sorted(words, key=lambda w: (w['coor_Y'], w['coor_X']))
        text = " ".join(word['text'] for word in sorted_words)
        block_texts[f"Block_{label}"] = text
    return block_texts

# === Process the resume PDF ===
def process_resume(pdf_path, n_clusters=8):
    images = convert_from_path(pdf_path, dpi=300)
    all_words = []
    for image in images:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        words = extract_words_with_coordinates(image_cv)
        all_words.extend(words)

    clustered_blocks = cluster_words_spatially(all_words, n_clusters=n_clusters)
    block_texts = assemble_blocks_from_clusters(clustered_blocks)

    print(json.dumps(block_texts, indent=4, ensure_ascii=False))

# === Run the full pipeline ===
if __name__ == "__main__":
    pdf_path = r"D:\Resume_QA\QUINIQUE\Vishal Rajpure_selected.pdf"  # Replace with your PDF path
    process_resume(pdf_path, n_clusters=8)