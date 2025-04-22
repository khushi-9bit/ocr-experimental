import json
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "textattack/roberta-base-CoLA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

nltk.download('punkt')
def load_text_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_text = " ".join([page["normal_text"] for page in data if "normal_text" in page and page["normal_text"]])
    return all_text

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

# Grammar score function
def grammar_acceptability_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0][1])  # Probability of being grammatically acceptable

# Process everything
def process_json_file(json_path, output_path="grammar_scores.txt"):
    text_from_json = load_text_from_json(json_path)
    sentence_chunks = chunk_by_sentences(text_from_json, max_sentences=4)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(sentence_chunks):
            score = grammar_acceptability_score(chunk)
            out.write(f"Chunk {i+1}:\n{chunk}\nScore: {score:.4f}\n\n")
            print(f"✅ Chunk {i+1} scored: {score:.4f}")
    
    print(f"\n📝 All grammar scores saved to '{output_path}'")

# === Run the full pipeline ===
if __name__ == "__main__":
    process_json_file("output.json", "grammar_scores.txt")
