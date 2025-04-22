# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Load FLAN-T5 (or T5)
# model_name = "google/flan-t5-large"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# def complete_paragraph(paragraph):
#     prompt = f"Complete this paragraph: {paragraph}"
#     inputs = tokenizer(prompt, return_tensors="pt")
    
#     output = model.generate(**inputs, max_length=200, num_return_sequences=1)
#     completed_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     print("\n🔹 Suggested Completion:")
#     print(completed_text)

# paragraph = "The sun rises in the east because of Earth's rotation. Every morning, people witness the sunrise."
# complete_paragraph(paragraph)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def find_missing_part(paragraph):
    hypothesis = "This paragraph expresses a full idea with a clear beginning, middle, and end."
    inputs = tokenizer(paragraph, hypothesis, return_tensors="pt", truncation=True, padding=True)

    # Get attention weights (indicating missing info)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Labels: [0] - Incomplete, [1] - Uncertain, [2] - Complete
    prediction = torch.argmax(probs).item()
    
    if prediction == 0:
        print("🔴 Paragraph is Incomplete!")
    elif prediction == 1:
        print("🟡 Paragraph is Uncertain!")
    else:
        print("🟢 Paragraph is Complete!")
    
    # Highlight missing tokens based on attention scores
    attentions = outputs.attentions[-1]  # Get last layer attention scores
    token_importance = torch.mean(attentions, dim=1).squeeze()  # Average across heads

    # Get important tokens (Low attention → Potentially missing)
    tokens = tokenizer.tokenize(paragraph)
    missing_tokens = [tokens[i] for i in range(len(tokens)) if token_importance[i] < 0.2]

    if missing_tokens:
        print("\n🔹 Missing or Weak Parts:")
        print(" ".join(missing_tokens))
    else:
        print("\n✅ No major missing parts detected.")

# Test Example
paragraph = "The sun rises in the east. This happens every morning."
find_missing_part(paragraph)


# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Load RoBERTa tokenizer and model
# MODEL_NAME = "roberta-large-mnli"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# # Function to check paragraph validity
# def check_paragraph_validity(paragraph):
#     # Define the hypothesis
#     hypothesis = "This paragraph is complete and valid."

#     # Tokenize input text
#     inputs = tokenizer(paragraph, hypothesis, return_tensors="pt", truncation=True, padding=True)
    
#     # Run inference
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Get prediction scores
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)

#     # Labels: [0] - Contradiction (Invalid), [1] - Neutral (Uncertain), [2] - Entailment (Valid)
#     labels = ["Incomplete", "Uncertain", "Complete"]
#     prediction = torch.argmax(probs).item()

#     # Print results
#     print(f"🔹 Paragraph: {paragraph}\n")
#     print(f"✅ Classification: {labels[prediction]}")
#     print(f"🟢 Complete: {probs[0][2]:.2f}, 🔵 Uncertain: {probs[0][1]:.2f}, 🔴 Incomplete: {probs[0][0]:.2f}")

# # Example paragraphs
# paragraph_1 = "Figure 9.1 shows the patterns of isobars corresponding to pressure systems. Low-pressure system is enclosed by one or more isobars with the lowest pressure in the centre.High-pressure system is also enclosed by one or more isobars with the highest pressure in the centre."
# paragraph_2 = "Frictional Force affects the speed of the wind. It is greatest at the surface and its influence generally extends upto an elevation of 1 - 3 km. Over the sea surface the friction is minimal."
# paragraph_3 = "You already know that the air is set in motion due to the differences in atmospheric pressure. The air in motion is called wind. The wind blows from high pressure to low pressure. The wind at the surface experiences friction. In addition, rotation of the earth also affects the wind movement. The force exerted by the rotation of the earth is known as the Coriolis force. Thus, the horizontal winds near the earth surface respond to the combined effect of three forces – the pressure gradient force, the frictional force and the Coriolis force. In addition, the gravitational force acts downwrd."

# print("\n--- Checking Paragraph 1 ---")
# check_paragraph_validity(paragraph_1)

# print("\n--- Checking Paragraph 2 ---")
# check_paragraph_validity(paragraph_2)

# print("\n--- Checking Paragraph 3 ---")
# check_paragraph_validity(paragraph_3)


# import fitz  # PyMuPDF
# from transformers import pipeline
# def extract_paragraphs_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     paragraphs = []

#     for page in doc:
#         text = page.get_text("text")  # Extract text
#         paragraphs.extend(text.split("\n\n"))  # Split by double newlines

#     return [para.strip() for para in paragraphs if para.strip()]

# segmenter = pipeline("text2text-generation", model="google/flan-t5-large")

# def chunk_text_transformer(text):
#     """Uses a transformer model to chunk text into meaningful segments."""
#     outputs = segmenter(text, max_length=512, truncation=True)
#     return outputs[0]['generated_text'].split("\n\n")  # Splitting into paragraphs

# pdf_path = r"C:\Users\KhushiOjha\Downloads\geo_chap_9.pdf"
# paragraphs = extract_paragraphs_from_pdf(pdf_path)

# print("Extracted Paragraphs:", paragraphs)

# if not paragraphs:
#     print("No text extracted from PDF. Check if the PDF has selectable text!")
# else:
#     text = "\n\n".join(paragraphs)
#     print("✅ Text Extracted Successfully!")

#     paragraph_chunks = chunk_text_transformer(text)

#     # ✅ Debug: Print what Flan-T5 generates
#     print("Generated Paragraphs:", paragraph_chunks)

# pdf_path = r"C:\Users\KhushiOjha\Downloads\geo_chap_9.pdf"
# paragraphs = extract_paragraphs_from_pdf(pdf_path)
# text = "\n\n".join(paragraphs)
# paragraph_chunks = chunk_text_transformer(text)
# print(paragraph_chunks)

