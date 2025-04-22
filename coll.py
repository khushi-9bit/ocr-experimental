# import re
# from transformers import pipeline

# # Sample resume text
# text = """
# He has experience with Python, Java, and SQL.
# """

# # --------- Step 1: Extract Skills with NER Model ---------
# ner_pipeline = pipeline("ner", model="ClaudiuFilip1100/ner-resume", aggregation_strategy="simple")
# ner_results = ner_pipeline(text)

# skills = [ent['word'] for ent in ner_results if ent['entity_group'] == 'SKILL']

# # Remove duplicates while keeping order
# unique_skills = list(dict.fromkeys(skills))

# # --------- Step 2: Extract Years of Experience ---------
# # This regex captures "X years", "X+ years", etc.
# experience_matches = re.findall(r'(\d+)\+?\s+years?', text.lower())
# total_years = sum(int(x) for x in experience_matches)

# # --------- Output ---------
# print("Extracted Skills:")
# print(unique_skills)

# print("\nTotal Estimated Experience:")
# print(f"{total_years} years")


# from transformers import pipeline

# # Load a generic NER pipeline
# ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# # Input text (resume or job description)
# text = """
# John has experience in Python, SQL, AWS, TensorFlow, and Docker. He also has a deep understanding of machine learning and data science.
# """

# # Get skill entities (SKILL is based on model's training)
# skills = [entity['word'] for entity in ner_pipeline(text) if entity['entity_group'] == 'SKILL']

# print("Extracted Skills:", skills)

# from transformers import pipeline

# # Try a more specific model for skill extraction
# ner_pipeline = pipeline("ner", model="talent/skills-extraction", aggregation_strategy="simple")

# # Sample text
# text = """
# John has experience in Python, SQL, AWS, TensorFlow, and Docker. He also has a deep understanding of machine learning and data science.
# """

# # Extract skills
# skills = [entity['word'] for entity in ner_pipeline(text) if entity['entity_group'] == 'SKILL']
# print("Extracted Skills:", skills)


# import re
# from transformers import pipeline
# from dateutil import parser
# from datetime import datetime

# # Load a general-purpose NER model
# ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# # Input text (replace with your resume text)
# text = """
# E D U C AT I O N
# National Institute of Technology, Jamshedpur
# 2020
# B. Tech in Computer Science and Engineering
# 7.73
# E X P E R I E N C E
# Genpact India Private Limited
# Oct-2020 - Mar-2023
# QA Engineer
# Project1: Pharmacovigilance Artificial Intelligence(PVAI)
# PVAI is an web application which is made with AI and ML module to automate the
# application to prevent the manual effort for to maintain the records of drug
# reaction.
# Verified desired functionality and performance according to business requirement
# and functional specification documents through manual and automated testing
# procedures.
# Created test scenarios, test cases and test data for validating features and
# functionality of the web application based on functional requirements.
# I have written automation test script for all possible scenario using selenium with
# Java to test and validate the requirements quickly and generated the test report
# using Allure Reports
# Executed all possible test scenarios and test cases in web application to sure the
# quality of the software and accordingly raised defects in the tool JIRA.
# Regularly tracking the build and based on the product deployment performed
# smoke, Regression,Functional, Formal and Dry Run testing.
# I have tested all possible feature and functionality using API Testing with the help
# of POSTMAN to make sure all the API are running correctly in the background.
# With the help of MariaDB i did database testing to sure the quality of data that are
# getting captured correctly in database or not.
# Genpact India Private Limited
# Apr-2023 - Present
# QA Engineer
# Project2: Pfizer SharePoint Project
# """

# ### 1. Extract entities like skills using NER
# def extract_entities(text):
#     entities = ner_pipeline(text)
#     skills = []
#     for ent in entities:
#         if ent['entity_group'] in ['MISC', 'ORG', 'SKILL']:  # You can tune this
#             if len(ent['word']) > 1:
#                 skills.append(ent['word'])
#     return list(set(skills))

# ### 2. Extract job/edu durations using regex
# def extract_experience_periods(text):
#     pattern = r"([A-Z][a-z]{2,9})\.?\s*(\d{2,4})\s*[–-]\s*([A-Z][a-z]{2,9})\.?\s*(\d{2,4})"
#     short_pattern = r"([A-Z]{3,})\s*(\d{2})\s*[–-]\s*([A-Z]{3,})\s*(\d{2})"

#     matches = re.findall(pattern, text) + re.findall(short_pattern, text)
#     durations = []

#     for match in matches:
#         try:
#             start = parser.parse(f"{match[0]} {match[1] if len(match[1]) == 4 else '20'+match[1]}")
#             end = parser.parse(f"{match[2]} {match[3] if len(match[3]) == 4 else '20'+match[3]}")
#             months = (end.year - start.year) * 12 + (end.month - start.month)
#             durations.append((start.strftime("%b %Y"), end.strftime("%b %Y"), months))
#         except Exception as e:
#             print("Error:", e)
#     return durations

# # Run functions
# skills = extract_entities(text)
# durations = extract_experience_periods(text)

# # Output
# print("Extracted Skills:", skills)
# for start, end, months in durations:
#     print(f"Experience Duration: {start} to {end} ({months} months)")

# from pyresparser import ResumeParser

# # Provide full path to your resume file
# data = ResumeParser("D:\Resume_QA\QUINIQUE\Raju Lohra.pdf").get_extracted_data()

# # Print the parsed data
# for key, value in data.items():
#     print(f"{key}: {value}")

from pyresparser import ResumeParser

# Path to your PDF resume
resume_path = r"D:\Ocr_image\Resume_QA\Raju Lohra.pdf"

# Parse the resume
data = ResumeParser(resume_path).get_extracted_data()

# Print results
if data:
    print("Name:", data.get("name"))
    print("Email:", data.get("email"))
    print("Phone:", data.get("mobile_number"))
    print("Skills:", data.get("skills"))
    print("Degree:", data.get("degree"))
    print("Education:", data.get("education"))
    print("Companies:", data.get("company_names"))
    print("Experience (in years):", data.get("total_experience"))
else:
    print("Failed to extract data.")
