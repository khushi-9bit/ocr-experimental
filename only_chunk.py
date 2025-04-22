# import json
# import nltk
# from nltk.tokenize import sent_tokenize

# nltk.download('punkt')

# def load_text_from_json(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     # Combine all "normal_text" fields
#     all_text = " ".join([page["normal_text"] for page in data if "normal_text" in page and page["normal_text"]])
#     return all_text

# def chunk_by_sentences(text, max_sentences=5):
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

# if __name__ == "__main__":
#     json_path = "output.json"
#     output_txt_path = "json_text_chunks.txt"

#     # Load and chunk
#     text_from_json = load_text_from_json(json_path)
#     sentence_chunks = chunk_by_sentences(text_from_json, max_sentences=4)

#     print("🧩 Sentence Chunks from JSON:")
#     with open(output_txt_path, "w", encoding="utf-8") as f:
#         for i, chunk in enumerate(sentence_chunks):
#             chunk_str = f"\n[{i+1}] {chunk}\n"
#             print(chunk_str)
#             f.write(chunk_str)

#     print(f"\n✅ All chunks saved to '{output_txt_path}'")


import json
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# === Grammar Model ===
grammar_model_name = "textattack/roberta-base-CoLA"
grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
grammar_model = AutoModelForSequenceClassification.from_pretrained(grammar_model_name)
grammar_model.eval()

# === Zero-shot Classification Model for Semantic Meaning ===
semantic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
semantic_labels = ["Meaningful", "Nonsense", "Unrelated"]

# === Download NLTK tokenizer ===
nltk.download('punkt')

# === Load JSON and extract text ===
def load_text_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_text = " ".join([page["normal_text"] for page in data if "normal_text" in page and page["normal_text"]])
    return all_text

# === Chunk into sentence groups ===
def chunk_by_sentences(text, max_sentences=4):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    for sentence in sentences:
        current_chunk.append(sentence.strip())
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# === Grammar score function ===
def grammar_acceptability_score(text):
    inputs = grammar_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = grammar_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0][1])  # Probability of being grammatically acceptable

# === Semantic Meaning Score ===
def semantic_meaning_score(text):
    result = semantic_classifier(text, candidate_labels=semantic_labels)
    return {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}

# === Process everything ===
def process_json_file(json_path, output_path="combined_scores.txt"):
    text_from_json = load_text_from_json(json_path)
    sentence_chunks = chunk_by_sentences(text_from_json, max_sentences=4)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(sentence_chunks):
            grammar_score = grammar_acceptability_score(chunk)
            semantic_score = semantic_meaning_score(chunk)
            out.write(f"Chunk {i+1}:\n{chunk}\n")
            out.write(f"Grammar Score: {grammar_score:.4f}\n")
            for label, score in semantic_score.items():
                out.write(f"{label} Score: {score:.4f}\n")
            out.write("\n")
            print(f"✅ Chunk {i+1} processed")

    print(f"\n📝 All scores saved to '{output_path}'")

# === Run the full pipeline ===
if __name__ == "__main__":
    process_json_file("output.json", "combined_scores.txt")


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # CoLA = Corpus of Linguistic Acceptability
# model_name = "textattack/roberta-base-CoLA"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# model.eval()

# def grammar_acceptability_score(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     probs = torch.softmax(logits, dim=-1)
#     # Class 1 = Acceptable (grammatically correct)
#     return float(probs[0][1])

# # Example
# text_1 = "The pressure in the centre is higher."
# text_2 = "Wooglefish crandled in blorptastic zone."

# print("Text 1 Grammar Score:", grammar_acceptability_score(text_1))
# print("Text 2 Grammar Score:", grammar_acceptability_score(text_2))
