import cohere

co = cohere.Client("qzrDlifnnIwUX9XGTfVwa4kRXd0zMylNTYurqKNO")  # Replace with your API key

# Define context documents (your clustered blocks or OCR content)
results = [
            "Vertical Variation of Pressure In the lower atmosphere the pressure decreases rapidly with height. The decrease amounts to about 1 mb for each 10 m increase in elevation. It does not always decrease at the same rate. Table 9.1 gives the average pressure and temperature at selected levels of elevation for a standard atmosphere.",
            "11092CHID CHAPTER arlier Chapter 8 described the uneven EK distribution of temperature over the surface of the earth. Air expands when heated and gets compressed when cooled. This results in variations in the atmospheric pressure. The result is that it causes the movement of air from high pressure to low pressure, setting the air in motion.",
            "Consult your book, Practical Work in Geography — Part I(NCERT, 2006) and learn about these instruments. The pressure decreases with height. At any elevation it varies from place to place and its variation is the primary cause of air motion, i.e. wind which moves from high pressure areas to low pressure areas."
 
]

# System prompt to limit answers strictly to the provided documents
system_prompt = "You are a helpful assistant. Use ONLY the provided context documents to answer questions. If the answer is not in the documents, say: 'The answer is not in the context.' Do not guess or hallucinate."

def generate_answer(query_text, context_blocks):
    try:
        response = co.chat(
            message=query_text,
            documents=[{"title": f"Block {i+1}", "snippet": text} for i, text in enumerate(context_blocks)],
            model="command-r-plus",
            chat_history=[
                {"role": "SYSTEM", "message": system_prompt}
            ]
        )
        print(f"\n✅ Answer: {response.text.strip()}")
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Interactive Q&A
if __name__ == "__main__":
    while True:
        question = input("\n❓ Your question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        generate_answer(question, results)
