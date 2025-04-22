import pdfplumber
import json

def extract_tables_and_text(pdf_path):
    extracted_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_content = {"page_number": page_num + 1, "text": "", "tables": []}
            
            # Extract text
            text = page.extract_text()
            if text:
                page_content["text"] = text.strip()

            # Extract tables with better structure
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table and len(table) > 1:  # Ensure it's not empty/misidentified
                        headers = table[0]  # First row as headers
                        structured_table = []

                        for row in table[1:]:  # Process the data rows
                            row_dict = {}
                            for i in range(len(headers)):
                                if headers[i] and row[i]:  # Avoid None values
                                    row_dict[headers[i].strip()] = row[i].strip()
                            structured_table.append(row_dict)

                        if structured_table:
                            page_content["tables"].append(structured_table)

            extracted_data.append(page_content)

    return extracted_data

# Example usage
pdf_path = r"C:\Users\KhushiOjha\Downloads\ilovepdf_split(1)\geo_pdf-1-1.pdf"
output_data = extract_tables_and_text(pdf_path)

# Save to JSON
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("✅ Extraction Complete. Data saved to output.json")
