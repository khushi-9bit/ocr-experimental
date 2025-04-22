# import spacy
# import re
# import json

# # Load spaCy English model
# nlp = spacy.load("en_core_web_sm")

# # --- Regex Patterns ---
# EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
# PHONE_REGEX = r"\+?\d[\d\s\-()]{9,}\d"
# YEAR_RANGE_REGEX = r"(19|20)\d{2}\s*[-–—to]{1,3}\s*(19|20)\d{2}"

# # --- NER and Regex Extractor ---
# def extract_entities(text):
#     doc = nlp(text)
#     return {
#         "names": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
#         "orgs": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
#         "locations": [ent.text for ent in doc.ents if ent.label_ == "GPE"],
#         "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
#         "emails": re.findall(EMAIL_REGEX, text),
#         "phones": re.findall(PHONE_REGEX, text),
#         "year_ranges": [f"{a}-{b}" for a, b in re.findall(YEAR_RANGE_REGEX, text)],
#     }

# # --- Section Extractor ---
# def extract_section(text, keywords):
#     for key in keywords:
#         if key in text:
#             start = text.find(key)
#             return text[start:start + 1000]  # Extract chunk after heading
#     return ""

# def guess_sections(text):
#     lower = text.lower()
#     return {
#         "education_section": extract_section(lower, ["education", "qualification"]),
#         "experience_section": extract_section(lower, ["experience", "employment", "work"]),
#     }

# # --- Main Processing Logic ---
# def process_resume(text):
#     all_entities = extract_entities(text)
#     sections = guess_sections(text)

#     return {
#         "personal_info": {
#             "name": all_entities["names"][0] if all_entities["names"] else "",
#             "email": all_entities["emails"][0] if all_entities["emails"] else "",
#             "phone": all_entities["phones"][0] if all_entities["phones"] else "",
#             "location": all_entities["locations"][0] if all_entities["locations"] else "",
#         },
#         "education": {
#             "text": sections["education_section"],
#             "periods": all_entities["year_ranges"]
#         },
#         "experience": {
#             "text": sections["experience_section"],
#             "companies": all_entities["orgs"],
#             "periods": all_entities["year_ranges"]
#         }
#     }

# # --- Test it with sample resume text ---
# if __name__ == "__main__":
#     sample_text = """
#     84 Thunderstorms and Tornadoes Other severe local storms are thunderstorms and tornadoes. They are of short duration, occurring over a small area but are violent. Thunderstorms are caused by intense convection on moist hot days. A thunderstorm is a well-grown cumulonimbus cloud producing thunder and lightening. When the clouds extend to heights where sub-zero temperature prevails, hails are formed and they come down as hailstorm. If there is insufficient moisture, a thunderstorm can generate dust- storms. A thunderstorm is characterised by intense updraft of rising warm air, which causes the clouds to grow bigger and rise to FUNDAMENTALS OF PHYSICAL GEOGRAPHY greater height. This causes precipitation. Later, downdraft brings down to earth the cool air and the rain. From severe thunderstorms sometimes spiralling wind descends like a trunk of an elephant with great force, with very low pressure at the centre, causing massive destruction on its way. Such a phenomenon is called a tornado. Tornadoes generally occur in middle latitudes. The tornado over the sea is called water spouts. These violent storms are the manifestation of the atmosphere’s adjustments to varying energy distribution. The potential and heat energies are converted into kinetic energy in these storms and the restless atmosphere again returns to its stable state. EXERCISES 1. Multiple choice questions. (i) If the surface air pressure is 1,000 mb, the air pressure at 1 km above the surface will be: (a) 700 mb (b) 1,100 mb (c) 900 mb (d) 1,300 mb (ii) The Inter Tropical Convergence Zone normally occurs: (a) near the Equator (c) near the Tropic of Capricorn (b) near the Tropic of Cancer (d) near the Arctic Circle (iii) The direction of wind around a low pressure in northern hemisphere is: (a) clockwise (b) perpendicular to isobars (c) anti-clock wise (d) parallel to isobars (iv) Which one of the following is the source region for the formation of air masses? (a) the Equatorial forest (b) the Himalayas (c) the Siberian Plain (d) the Deccan Plateau 2. Answer the following questions in about 30 words. (i) What is the unit used in measuring pressure? Why is the pressure measured at station level reduced to the sea level in preparation of weather maps? (ii) While the pressure gradient force is from north to south, i.e. from the subtropical high pressure to the equator in the northern hemisphere, why are the winds north easterlies in the tropics. (iii) What are the geotrophic winds? (iv) Explain the land and sea breezes. 2024-25
#     """

#     print("🔍 Processing sample resume...")
#     structured_data = process_resume(sample_text)

#     print("\n📄 Extracted Structured Resume Data:")
#     print(json.dumps(structured_data, indent=4))

#     # Save to file
#     with open("structured_resume.json", "w", encoding="utf-8") as f:
#         json.dump(structured_data, f, indent=4, ensure_ascii=False)


# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# import json
# import spacy
# from pdf2image import convert_from_path
# import nltk
# from nltk.tokenize import sent_tokenize

# nltk.download("punkt")
# nlp = spacy.load("en_core_web_sm")  

# from transformers import pipeline

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# labels = ["Education", "Skills", "Experience", "Projects", "Personal Details", "Certificates", "Other"]

# def classify_paragraph_zero_shot(text):
#     result = classifier(text, labels)
#     return result["labels"][0]  # Most likely category

# def detect_borders_opencv(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     border_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
#     return border_boxes

# def extract_text_with_ocr(image):
#     custom_config = r'--oem 3 --psm 12'
#     ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
#     words = [ocr_data["text"][i] for i in range(len(ocr_data["text"])) if ocr_data["text"][i].strip()]
#     return " ".join(words)

# def extract_named_entities(text):
#     doc = nlp(text)
#     return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

# def chunk_text_into_paragraphs(text, sentences_per_chunk=3):
#     sentences = sent_tokenize(text)
#     paragraphs = []
#     for i in range(0, len(sentences), sentences_per_chunk):
#         paragraph = " ".join(sentences[i:i + sentences_per_chunk])
#         if paragraph.strip():
#             paragraphs.append(paragraph.strip())
#     return paragraphs
# def extract_data_from_pdf(pdf_path):
#     categorized_output = {label: [] for label in labels}
#     images = convert_from_path(pdf_path, dpi=300)

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
#             image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             borders = detect_borders_opencv(image_cv)

#             full_text = extract_text_with_ocr(image_cv)
#             paragraph_chunks = chunk_text_into_paragraphs(full_text)

#             for para in paragraph_chunks:
#                 category = classify_paragraph_zero_shot(para)
#                 categorized_output[category].append(para)

#     return categorized_output

# if __name__ == "__main__":
#     pdf_path = r"D:\Resume_QA\QUINIQUE\Vishal Rajpure_selected.pdf"

#     print("🔍 Extracting and classifying PDF paragraphs...")
#     output_data = extract_data_from_pdf(pdf_path)

#     print("💾 Saving categorized output to output_by_category.json...")
#     with open("output_by_category.json", "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=4, ensure_ascii=False)


import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.tokenize import sent_tokenize 
import nltk
import json

nltk.download("punkt")

# Models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["Education", "Skills", "Experience", "Projects", "Personal Details", "Certificates", "Other"]

def extract_words_with_coords(image_cv):
    data = pytesseract.image_to_data(image_cv, config='--psm 6', output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        word_info = {
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i]
        }
        words.append(word_info)
    return words

def group_words_into_paragraphs(words, y_threshold=15):
    words.sort(key=lambda x: x["top"])
    paragraphs = []
    current_para = []
    prev_y = None

    for word in words:
        if prev_y is None or abs(word["top"] - prev_y) <= y_threshold:
            current_para.append(word)
        else:
            if current_para:
                para_text = " ".join([w["text"] for w in current_para])
                avg_x = int(np.mean([w["left"] for w in current_para]))
                avg_y = int(np.mean([w["top"] for w in current_para]))
                paragraphs.append({"text": para_text, "x": avg_x, "y": avg_y})
            current_para = [word]
        prev_y = word["top"]

    if current_para:
        para_text = " ".join([w["text"] for w in current_para])
        avg_x = int(np.mean([w["left"] for w in current_para]))
        avg_y = int(np.mean([w["top"] for w in current_para]))
        paragraphs.append({"text": para_text, "x": avg_x, "y": avg_y})

    return paragraphs

def cluster_paragraphs(paragraphs, k=6):
    vectors = []
    for para in paragraphs:
        emb = embedder.encode(para["text"])
        vector = np.append(emb, [para["x"]/1000, para["y"]/1000])  # normalize coords
        vectors.append(vector)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_pred = kmeans.fit_predict(vectors)

    clustered = {i: [] for i in range(k)}
    for idx, label in enumerate(labels_pred):
        clustered[label].append(paragraphs[idx])

    classified_clusters = {label: [] for label in labels}
    for cluster_id, paras in clustered.items():
        cluster_text = " ".join([p["text"] for p in paras[:3]])  # sample few
        predicted = classifier(cluster_text, labels)
        cat = predicted["labels"][0]
        classified_clusters[cat].extend([p["text"] for p in paras])

    return classified_clusters

def process_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    all_paragraphs = []
    for image in images:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        words = extract_words_with_coords(image_cv)
        paras = group_words_into_paragraphs(words)
        all_paragraphs.extend(paras)

    final_data = cluster_paragraphs(all_paragraphs)
    return final_data

if __name__ == "__main__":
    pdf_path = r"D:\Resume_QA\QUINIQUE\Vishal Rajpure_selected.pdf"
    print("📄 Processing resume with semantic + layout clustering...")
    output = process_pdf(pdf_path)

    with open("clustered_resume_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print("✅ Output saved to clustered_resume_output.json")


