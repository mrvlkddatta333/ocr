import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import spacy

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_image(img):
    # Convert image to grayscale if it's not already
    if len(np.array(img).shape) == 3:
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    else:
        gray_img = np.array(img)
    # Apply thresholding
    _, thresh_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh_img

def detect_and_recognize_text(img, lang='eng'):
    # Preprocess the image for text recognition
    preprocessed_img = preprocess_image(img)

    # Use Tesseract to detect text boxes on the preprocessed image
    boxes = pytesseract.image_to_boxes(preprocessed_img, lang=lang)

    # Draw bounding boxes on a copy of the original image (for display)
    display_img = np.array(img).copy()
    # print(display_img.shape)
    if len(display_img.shape) == 2:  # Grayscale image
        h, w = display_img.shape
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing boxes
    else:  # Color image
        h, w, _ = display_img.shape

    for b in boxes.splitlines():
        b = b.split(' ')
        display_img = cv2.rectangle(display_img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # Use Tesseract to recognize text from the preprocessed image
    custom_config = f'--oem 3 --psm 6 -l {lang}'
    detected_text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

    return detected_text, display_img

def analyze_text(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract tokens, part-of-speech tags, and named entities
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return tokens, pos_tags, entities

def main():
    st.title("OCRius - Intelligent Visual Text Extraction and Recognition Engine")

    # Language selection
    language_options = {
        'English': 'eng',
        'French': 'fra',
        'German': 'deu',
        'Spanish': 'spa',
        'Telugu': 'tel',
        'Hindi': 'hin',
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
                detected_text, annotated_img = detect_and_recognize_text(img, lang=language_options[selected_language])
                st.success("Text extracted successfully!")

            # Display the detected text and annotated image
            st.subheader("Extracted Text")
            st.write(detected_text)
            # st.subheader("Annotated Image")
            # st.image(annotated_img, use_column_width=True)

            # Analyze the text with NLP
            tokens, pos_tags, entities = analyze_text(detected_text)

            # Display NLP analysis results
            st.subheader("NLP Analysis")
            st.write("Tokens:", tokens)
            st.write("Part-of-Speech Tags:", pos_tags)
            st.write("Named Entities:", entities)

if __name__ == "__main__":
    main()
