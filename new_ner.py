import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

def show_entities(text):
    doc = nlp(text)
    if not doc.ents:
        print("❌ No named entities found in the text.")
    else:
        print("✅ Named Entities Found:\n")
        for ent in doc.ents:
            print(f"{ent.text:<30} --> {ent.label_}")

if __name__ == "__main__":
    print("🔍 Enter text to analyze for named entities:\n")
    user_input = input("Text: ")
    print("\n--- Entity Results ---\n")
    show_entities(user_input)
