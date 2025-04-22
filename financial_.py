import os
import cv2
import numpy as np
import torch
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

def load_model():
    tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForTokenClassification.from_pretrained("nielsr/layoutlm-finetuned-publaynet")
    return tokenizer, model

def resize_image(image, max_size=1024):
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    return image

def detect_tables(image, processor, model, output_folder, page_num):
    image = Image.fromarray(image)
    image = resize_image(image)
    encoding = processor(image, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    
    if 1 in predictions:  # 1 corresponds to 'table' class
        cv2.imwrite(os.path.join(output_folder, f'table_page_{page_num}.png'), np.array(image))

def process_pdf(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    processor, model = load_model()
    
    for i, image in enumerate(images):
        image_np = np.array(image)
        detect_tables(image_np, processor, model, output_folder, i)

if __name__ == "__main__":
    pdf_path = r"C:\Users\KhushiOjha\Downloads\finance2.pdf"  # Change this to your PDF file path
    output_folder = "output_tables"
    process_pdf(pdf_path, output_folder)
    print("Table detection complete. Check the output_tables folder.")
