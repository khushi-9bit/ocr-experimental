import json
from transformers import pipeline

# === Load OCR output JSON ===
def load_resume_text(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_text = ""
    for page in data:
        if "normal_text" in page:
            full_text += page["normal_text"] + " "
        if "bordered_text" in page:
            full_text += page["bordered_text"] + " "
    return full_text.strip()

# === Setup DistilBERT Q&A Pipeline ===
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# === Ask a Question ===
def ask_question(context, question):
    result = qa_model(question=question, context=context)
    return result["answer"]

# === Main Q&A Function ===
if __name__ == "__main__":
    context_text = load_resume_text("output.json")

    while True:
        user_question = input("\n❓ Ask a question (or type 'exit'): ")
        if user_question.lower() == "exit":
            break
        answer = ask_question(context_text, user_question)
        print("✅ Answer:", answer)
