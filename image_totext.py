import cv2
import pytesseract
import numpy as np

# Optional: Set the path to Tesseract executable if it's on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)

    # Step 2: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Apply Otsu's thresholding or adaptive thresholding for better contrast
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # alternative: adaptive thresholding
    # binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Step 5: Display the final image (binary image after preprocessing)
    cv2.imshow("Preprocessed Image", binary_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the window after the key press

    # Step 6: Apply OCR on the binary image with custom configurations
    custom_config = r'--oem 3 --psm 6'  # Adjust OCR settings for better results
    extracted_text = pytesseract.image_to_string(binary_image, config=custom_config)

    return extracted_text

# Path to your image
image_path = r"C:\Users\KhushiOjha\Downloads\Grayscale on Transparent.png"  # Update with the actual path to your image

# Extract text from the image
extracted_text = extract_text_from_image(image_path)

# Print the extracted text
print("Extracted Text:")
print(extracted_text)

"""
import cv2
import pytesseract
import numpy as np

# Optional: Set the path to Tesseract executable if it's on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)

    # Step 2: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Apply adaptive thresholding for better contrast (instead of Otsu's)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Optional: Show the binary image for debugging
    cv2.imshow("Preprocessed Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 5: Apply OCR on the binary image with custom configurations
    custom_config = r'--oem 3 --psm 6'  # Adjust OCR settings for better results
    extracted_text = pytesseract.image_to_string(binary_image, config=custom_config)

    return extracted_text

# Path to your image
image_path = r"C:\Users\KhushiOjha\Downloads\Grayscale on Transparent.png"  # Update with the actual path to your image

# Extract text from the image
extracted_text = extract_text_from_image(image_path)

# Print the extracted text
print("Extracted Text:")
print(extracted_text)

"""