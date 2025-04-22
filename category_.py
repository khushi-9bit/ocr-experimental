from transformers import pipeline
import json

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # Custom categories and their descriptions
# labels = {
#     "academic_details": "Education, degrees, certifications, universities",
#     "professional_details": "Work experience, job roles, companies, responsibilities",
#     "skill_details": "Programming languages, tools, frameworks, technologies",
#     "personal_details": "Contact info, email, phone number, objective",
#     "other_details": "Miscellaneous or unclassified information"
# }

label1 = {
    "academic_details": "This section includes any information related to education such as degrees earned, certifications, academic qualifications, schools, colleges, and universities attended.",
}
label2 = {
        "professional_details": "This section includes details about work experience, job titles, responsibilities, internships, company names, and duration of employment.",
}
label3 = {
        "skill_details": "This section includes technical and non-technical skills such as programming languages, tools, software, technologies, testing methods, and frameworks.",
}
label4 = {
        "personal_details": "This section includes personal contact details like full name, phone number, email address, personal summary or objective statement.",
}
label5 = {
        "other_details": "This section includes any other information that does not clearly belong to the previous categories, such as general statements, miscellaneous content, or unclear data."
}

# Reverse map from description to category key
description_to_key = {v: k for k, v in label2.items()}
candidate_labels = list(label2.values())

# Load resume chunks
with open("output.json", "r", encoding="utf-8") as f:
    resume_chunks = json.load(f)

# Output structure
grouped_chunks = {k: [] for k in label2.keys()}

# Classify each chunk
for chunk in resume_chunks:
    result = classifier(chunk, candidate_labels, multi_label=False)
    top_label_desc = result['labels'][0]
    score = result['scores'][0]

    if score > 0.2:
        top_key = description_to_key.get(top_label_desc, "other_details")
        grouped_chunks[top_key].append(chunk)

        #grouped_chunks["other_details"].append(chunk)

# Save output
with open("categorized_output.json", "w", encoding="utf-8") as f:
    json.dump(grouped_chunks, f, indent=2, ensure_ascii=False)

print(json.dumps(grouped_chunks, indent=2, ensure_ascii=False))



