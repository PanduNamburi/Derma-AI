import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- 1. DISEASE KNOWLEDGE BASE ---
DISEASE_INFO = {
    "Acne": {
        "info": "A common skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
        "tips": ["Wash your face twice a day with a mild cleanser.", "Avoid picking or squeezing pimples.",
                 "Use non-comedogenic makeup."]
    },
    "Actinic Keratosis": {
        "info": "A rough, scaly patch on the skin caused by years of sun exposure.",
        "tips": ["Protect your skin from UV rays using sunscreen.", "Wear protective clothing.",
                 "See a dermatologist as these can turn into cancer."]
    },
    "Benign Tumors": {
        "info": "Non-cancerous growths like moles or cysts that do not spread to other parts of the body.",
        "tips": ["Monitor for changes in size or color.", "Consult a doctor if the growth becomes painful."]
    },
    "Candidiasis": {
        "info": "A fungal infection caused by yeast, often appearing in warm, moist areas of the skin.",
        "tips": ["Keep the affected area dry and clean.", "Wear loose-fitting cotton clothing.",
                 "Antifungal creams are often helpful."]
    },
    "Eczema": {
        "info": "A condition that makes your skin red and itchy. It is common in children but can occur at any age.",
        "tips": ["Moisturize your skin at least twice a day.", "Identify and avoid triggers like harsh soaps.",
                 "Take shorter, lukewarm baths."]
    },
    "Psoriasis": {
        "info": "A skin disease that causes red, itchy scaly patches, most commonly on the knees and elbows.",
        "tips": ["Keep skin moisturized.", "Avoid skin injuries and stress.",
                 "Consult a doctor about specialized ointments."]
    },
    "Rosacea": {
        "info": "A condition that causes redness and often small, red, pus-filled bumps on the face.",
        "tips": ["Identify and avoid triggers like spicy foods.", "Apply sunscreen daily.",
                 "Use gentle skin care products."]
    },
    "Seborrheic Keratoses": {
        "info": "A noncancerous skin growth common in older adults. It usually looks waxy or wart-like.",
        "tips": ["Generally harmless and requires no treatment.", "Avoid scratching or rubbing the growth."]
    },
    "Skin Cancer": {
        "info": "The out-of-control growth of abnormal skin cells. Early detection is critical.",
        "tips": ["**Urgent:** Schedule an appointment with a dermatologist immediately.",
                 "Do not attempt home remedies.", "Avoid further sun exposure."]
    },
    "Vitiligo": {
        "info": "A disease that causes loss of skin color in patches.",
        "tips": ["Use sunscreen to protect depigmented patches.", "Consult a doctor about light therapy options."]
    },
    "Warts": {
        "info": "Small, grainy skin growths caused by the human papillomavirus (HPV).",
        "tips": ["Avoid sharing towels or razors.", "Do not pick at warts.",
                 "Salicylic acid treatments are often used."]
    }
}

# --- 2. CONFIGURATION & UI ---
st.set_page_config(page_title="SkinSense AI", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #455A64; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 SkinSense AI Analyzer")
st.write("Upload or capture a photo for an AI-powered dermatological analysis.")


# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="skin_model_final.tflite")
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_model()
labels = sorted(list(DISEASE_INFO.keys()))  # Ensures alphabetical matching with training


# --- 4. SKIN TEXTURE GATEKEEPER ---
def is_valid_skin_image(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Color Detection (HSV Filter)
    hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    skin_pct = (cv2.countNonZero(mask) / (img_cv.shape[0] * img_cv.shape[1])) * 100

    # Texture Detection (Laplacian Variance)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Thresholds: > 20% skin color and > 40 texture variance
    return skin_pct > 20 and texture_score > 40


# --- 5. MAIN APP LOGIC ---
st.subheader("Step 1: Capture Image")
picture = st.camera_input("Take a photo of the skin area")
uploaded_file = st.file_uploader("Or upload from gallery", type=["jpg", "jpeg", "png"])

source = picture if picture else uploaded_file

if source is not None:
    image = Image.open(source)
    st.image(image, caption='Image for Analysis', use_container_width=True)

    # Gatekeeper check
    if is_valid_skin_image(image):
        with st.spinner('Analyzing patterns...'):
            # Pre-processing
            img_array = np.array(image.convert('RGB'))
            img_resized = cv2.resize(img_array, (300, 300))
            input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)

            # Inference
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            # Results Processing
            idx = np.argmax(predictions)
            score = predictions[idx] * 100
            condition = labels[idx]

            st.divider()
            st.header(f"Result: {condition}")
            st.info(f"AI Confidence Score: {score:.2f}%")

            # Show Disease Info & Tips
            if condition in DISEASE_INFO:
                st.subheader(f"About {condition}")
                st.write(DISEASE_INFO[condition]["info"])

                st.subheader("💡 Care Recommendations")
                for tip in DISEASE_INFO[condition]["tips"]:
                    st.write(f"- {tip}")

            st.warning(
                "⚠️ **Disclaimer:** This tool is for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.")
    else:
        st.error(
            "❌ **Invalid Image:** The AI could not detect human skin texture. Please provide a clear, close-up photo.")

st.sidebar.title("About Project")
st.sidebar.info("Developed using EfficientNetB3 architecture to classify 11 categories of skin conditions.")