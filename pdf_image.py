import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np
import io
from PIL import Image  # Import Image from PIL

# Optional: Set the path to Tesseract executable if it's on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []

    # Iterate through each page of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Extract images from the page
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]  # Get the image reference
            base_image = doc.extract_image(xref)  # Extract image
            image_bytes = base_image["image"]  # Get image bytes
            image = Image.open(io.BytesIO(image_bytes))  # Convert bytes to image using PIL Image
            images.append(np.array(image))  # Convert the image to NumPy array

    return images

def extract_text_from_image(image):
    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 3: Apply adaptive thresholding for better contrast (instead of Otsu's)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Optional: Show the binary image for debugging
    cv2.imshow("Preprocessed Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 4: Apply OCR on the binary image with custom configurations
    custom_config = r'--oem 3 --psm 6'  # Adjust OCR settings for better results
    extracted_text = pytesseract.image_to_string(binary_image, config=custom_config)

    return extracted_text

# Path to your PDF
pdf_path = r"C:\Users\KhushiOjha\Downloads\Resume_Dev\Resume_Dev\Akshat-J.pdf"  # Update with the actual path to your PDF

# Extract images from the PDF
images = extract_images_from_pdf(pdf_path)

# Process each extracted image and extract text
for idx, image in enumerate(images):
    print(f"Extracting text from image {idx + 1}...")
    extracted_text = extract_text_from_image(image)
    print(f"Extracted Text from Image {idx + 1}:")
    print(extracted_text)
    print("-" * 80)
