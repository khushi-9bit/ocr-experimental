import pdfplumber
import re
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import MarkdownHeaderTextSplitter

nltk.download('punkt')


def extract_text_from_pdf(pdf_path):
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n".join(all_text)


def add_markdown_headers(text):
    markdown_lines = []
    lines = text.split('\n')

    for line in lines:
        stripped = line.strip()
        # Simple heuristic: ALL CAPS or Short line as a header
        if stripped.isupper() and 3 <= len(stripped.split()) <= 10:
            markdown_lines.append(f"# {stripped}")  # You can vary levels with more logic
        elif len(stripped.split()) <= 4 and stripped.endswith(":"):
            markdown_lines.append(f"## {stripped}")
        else:
            markdown_lines.append(stripped)

    return "\n".join(markdown_lines)


from langchain.text_splitter import MarkdownHeaderTextSplitter

def split_with_markdown_header(text):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(text)
    return chunks



if __name__ == "__main__":
    pdf_path = r"C:\Users\KhushiOjha\Downloads\geo_chap_9.pdf"

    print("📄 Extracting text...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("🔍 Detecting headers and converting to markdown...")
    markdown_text = add_markdown_headers(raw_text)

    print("✂️ Splitting using MarkdownHeaderTextSplitter...")
    chunks = split_with_markdown_header(markdown_text)

    print("📦 Final Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n🔹 Chunk {i+1}")
        print("Metadata:", chunk.metadata)
        print("Content:", chunk.page_content[:], "..." if len(chunk.page_content) > 300 else "")
