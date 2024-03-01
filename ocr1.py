import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_image(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh_img

def detect_and_recognize_text(img, lang='eng'):
    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Use Tesseract to detect and recognize text
    custom_config = f'--oem 3 --psm 6 -l {lang}'
    detected_text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

    return detected_text

def main():
    st.title("OCRius - Intelligent Visual Text Extraction and Recognition Engine")

    # Language selection
    language_options = {
        'English': 'eng',
        'French': 'fra',
        'German': 'deu',
        'Spanish': 'spa',
        # Add more languages and their corresponding Tesseract language codes here
    }
    selected_language = st.selectbox("Select Language", list(language_options.keys()))

    # File uploader for images
    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        # Display the uploaded image
        img = load_image(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Detect and recognize text
        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                detected_text = detect_and_recognize_text(img, lang=language_options[selected_language])
                st.success("Text extracted successfully!")

            # Display the detected text
            st.subheader("Extracted Text")
            st.write(detected_text)

if __name__ == "__main__":
    main()
