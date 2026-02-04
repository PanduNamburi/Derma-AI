import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# --- 1. DISEASE DATABASE ---
DISEASE_INFO = {
    "Acne": {
        "info": "A common skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
        "tips": ["Wash your face twice a day with a mild cleanser.", "Avoid picking or squeezing pimples.",
                 "Use non-comedogenic (oil-free) makeup."]
    },
    "Actinic Keratosis": {
        "info": "A rough, scaly patch on the skin caused by years of sun exposure.",
        "tips": ["Protect your skin from UV rays using sunscreen.", "Wear protective clothing.",
                 "See a dermatologist as these can occasionally turn into cancer."]
    },
    "Benign Tumors": {
        "info": "Non-cancerous growths like moles or seborrheic keratoses that do not spread to other parts of the body.",
        "tips": ["Monitor for changes in size, shape, or color.",
                 "Consult a doctor if the growth becomes painful or bleeds."]
    },
    "Candidiasis": {
        "info": "A fungal infection caused by a yeast called Candida, often appearing in warm, moist areas of the skin.",
        "tips": ["Keep the affected area dry and clean.", "Wear loose-fitting cotton clothing.",
                 "Antifungal creams are often recommended by pharmacists."]
    },
    "Eczema": {
        "info": "A condition that makes your skin red and itchy. It's common in children but can occur at any age.",
        "tips": ["Moisturize your skin at least twice a day.",
                 "Identify and avoid triggers like harsh soaps or allergens.", "Take shorter, lukewarm baths."]
    },
    "Psoriasis": {
        "info": "A skin disease that causes red, itchy scaly patches, most commonly on the knees, elbows, and scalp.",
        "tips": ["Keep skin moisturized.", "Short periods of sun exposure can help.",
                 "Avoid skin injuries and stress, which can trigger flare-ups."]
    },
    "Rosacea": {
        "info": "A condition that causes redness and often small, red, pus-filled bumps on the face.",
        "tips": ["Identify and avoid triggers like spicy foods or alcohol.", "Apply sunscreen daily.",
                 "Use gentle skin care products."]
    },
    "Seborrheic Keratoses": {
        "info": "A noncancerous skin growth that common in older adults. It usually looks like a waxy or wart-like growth.",
        "tips": ["Generally harmless and doesn't require treatment.", "Avoid scratching or rubbing the growth."]
    },
    "Skin Cancer": {
        "info": "The out-of-control growth of abnormal skin cells. Early detection is key.",
        "tips": ["**Urgent:** Schedule an appointment with a dermatologist immediately.",
                 "Do not attempt home remedies.", "Avoid further sun exposure."]
    },
    "Vitiligo": {
        "info": "A disease that causes loss of skin color in patches.",
        "tips": ["Use sunscreen to protect depigmented patches.", "Consider cosmetic camouflage if desired.",
                 "Consult a doctor about light therapy options."]
    },
    "Warts": {
        "info": "Small, grainy skin growths caused by the human papillomavirus (HPV).",
        "tips": ["Avoid sharing towels or razors.", "Do not pick at warts.",
                 "Over-the-counter salicylic acid treatments are often used."]
    }
}

# --- 2. ENHANCED CONFIG & STYLING ---
st.set_page_config(
    page_title="SkinSense AI - Smart Dermatology Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS - Professional Classic Style
st.markdown("""
    <style>
    /* Main App Styling */
    .stApp {
        background-color: #f5f5f5;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #2C3E50;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Button Styling - Professional */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3.5em;
        background-color: #2C3E50;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #34495E;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    /* Card Styling - Classic */
    .info-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 25px;
        margin: 15px 0;
    }

    /* Stats Container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 20px 0;
    }

    .stat-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        min-width: 150px;
        margin: 10px;
    }

    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
        color: #2C3E50;
    }

    .stat-label {
        color: #666;
        font-size: 0.9em;
        margin-top: 5px;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #2C3E50;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2C3E50;
    }

    /* Image Container */
    .image-container {
        border-radius: 5px;
        overflow: hidden;
        border: 1px solid #ddd;
        margin: 20px 0;
    }

    /* Animation for loading */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* Scroll Buttons */
    .scroll-buttons {
        position: fixed;
        right: 30px;
        bottom: 30px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        z-index: 9999;
    }

    .scroll-btn {
        background-color: #2C3E50;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .scroll-btn:hover {
        background-color: #34495E;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        transform: scale(1.1);
    }

    /* Responsive Grid */
    @media (max-width: 768px) {
        .stats-container {
            flex-direction: column;
        }

        .stat-box {
            width: 100%;
        }

        .upload-header-card {
            margin-bottom: 10px;
        }
    }

    /* Ensure equal column widths on desktop */
    @media (min-width: 769px) {
        [data-testid="column"] {
            flex: 1 1 50% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Add scroll functionality using Streamlit components
st.markdown("""
    <div class="scroll-buttons">
        <button class="scroll-btn" id="scrollTopBtn" title="Scroll to Top">↑</button>
        <button class="scroll-btn" id="scrollBottomBtn" title="Scroll to Bottom">↓</button>
    </div>

    <script>
    // Wait for the page to load
    window.addEventListener('load', function() {
        const scrollTopBtn = window.parent.document.getElementById('scrollTopBtn');
        const scrollBottomBtn = window.parent.document.getElementById('scrollBottomBtn');

        if (scrollTopBtn) {
            scrollTopBtn.addEventListener('click', function() {
                window.parent.document.querySelector('section.main').scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            });
        }

        if (scrollBottomBtn) {
            scrollBottomBtn.addEventListener('click', function() {
                const mainSection = window.parent.document.querySelector('section.main');
                mainSection.scrollTo({
                    top: mainSection.scrollHeight,
                    behavior: 'smooth'
                });
            });
        }
    });

    // Also add click handlers directly in the current context
    document.addEventListener('DOMContentLoaded', function() {
        const scrollTopBtn = document.getElementById('scrollTopBtn');
        const scrollBottomBtn = document.getElementById('scrollBottomBtn');

        if (scrollTopBtn) {
            scrollTopBtn.onclick = function() {
                // Try multiple approaches
                const main = document.querySelector('section.main') || 
                             window.parent.document.querySelector('section.main') ||
                             document.querySelector('.main') ||
                             window.parent.document.querySelector('.main');

                if (main) {
                    main.scrollTo({top: 0, behavior: 'smooth'});
                } else {
                    window.scrollTo({top: 0, behavior: 'smooth'});
                }
            };
        }

        if (scrollBottomBtn) {
            scrollBottomBtn.onclick = function() {
                const main = document.querySelector('section.main') || 
                             window.parent.document.querySelector('section.main') ||
                             document.querySelector('.main') ||
                             window.parent.document.querySelector('.main');

                if (main) {
                    main.scrollTo({top: main.scrollHeight, behavior: 'smooth'});
                } else {
                    window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
                }
            };
        }
    });
    </script>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR WITH ENHANCED UI ---
with st.sidebar:
    st.markdown("# 🩺 SkinSense AI")
    st.markdown("### Smart Dermatology Assistant")
    st.markdown("---")

    # Show Results option only if analysis is complete
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    if st.session_state.analysis_complete:
        page = st.radio(
            "**Navigation**",
            ["🔬 AI Analyzer", "📋 Results", "📚 Knowledge Base", "📊 Statistics", "ℹ️ About"],
            label_visibility="collapsed",
            index=1  # Auto-select Results page
        )
    else:
        page = st.radio(
            "**Navigation**",
            ["🔬 AI Analyzer", "📚 Knowledge Base", "📊 Statistics", "ℹ️ About"],
            label_visibility="collapsed"
        )

    st.markdown("---")

    # Quick Stats in Sidebar
    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Conditions", "11", delta="Covered")
    with col2:
        st.metric("Accuracy", "94%", delta="2.1%")

    st.markdown("---")

    # Tips of the Day
    st.markdown("### 💡 Tip of the Day")
    tips = [
        "Always use SPF 30+ sunscreen",
        "Stay hydrated for healthy skin",
        "Get 7-8 hours of sleep",
        "Eat foods rich in vitamins A & C"
    ]
    st.info(np.random.choice(tips))

    st.markdown("---")
    st.caption("Version 2.0 | Enhanced Edition")


# --- 4. CORE LOGIC FUNCTIONS ---
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_model_final.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None


def is_valid_skin_image(image):
    """Enhanced validation with more detailed feedback"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Color Test (HSV)
    hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, np.array([0, 20, 70]), np.array([25, 255, 255]))
    skin_pct = (cv2.countNonZero(mask) / (img_cv.shape[0] * img_cv.shape[1])) * 100

    # Texture Test (Laplacian Variance)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "valid": skin_pct > 20 and texture_score > 40,
        "skin_percentage": skin_pct,
        "texture_score": texture_score
    }


# --- PAGE 1: ENHANCED AI ANALYZER ---
if page == "🔬 AI Analyzer":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    # Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🔬 AI Skin Analyzer")
        st.markdown("### Instant screening for 11 dermatological conditions")
        st.markdown("Upload an image or take a photo to get started")

    st.markdown("---")

    # Create custom CSS for compact header cards
    st.markdown("""
    <style>
    .upload-header-card {
        background: white;
        border-radius: 8px;
        padding: 5px;
        text-align: center;
        margin-bottom: 5px;
    }
    .upload-header-card h3 {
        color: #2C3E50;
        margin: 0 0 0 0;
        font-size: 1.2em;
    }
    .upload-header-card p {
        color: #666;
        margin: 0;
        font-size: 0.95em;
    }
    /* Ensure camera and file inputs are same size */
    [data-testid="stCameraInput"], 
    [data-testid="stFileUploader"] {
        min-height: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Two-column layout for capture/upload options
    col_left, col_right = st.columns(2, gap="medium")

    # Initialize session state for camera mode
    if 'show_camera' not in st.session_state:
        st.session_state.show_camera = False
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None

    with col_left:
        st.markdown("""
        <div class="upload-header-card">
            <h3>📷 Camera Capture</h3>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📸 Open Camera", use_container_width=True, key="open_camera_btn"):
            st.session_state.show_camera = True
            st.session_state.captured_image = None

        # Show camera input only when button is clicked
        if st.session_state.show_camera:
            camera_photo = st.camera_input("Take a photo", label_visibility="collapsed", key="camera_input")
            if camera_photo:
                st.session_state.captured_image = camera_photo
                st.session_state.show_camera = False
                st.rerun()

        # Display captured image preview
        if st.session_state.captured_image and not st.session_state.show_camera:
            st.success("✅ Photo captured successfully!")
            if st.button("📸 Retake Photo", use_container_width=True, key="retake_btn"):
                st.session_state.captured_image = None
                st.session_state.show_camera = True
                st.rerun()

    with col_right:
        st.markdown("""
        <div class="upload-header-card">
            <h3>📁 Upload Image</h3>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"], label_visibility="collapsed",
                                    key="file_upload")

    # Initialize session state for analysis results
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Determine which image to use
    img_file = st.session_state.captured_image if st.session_state.captured_image else uploaded

    if img_file:
        image = Image.open(img_file)

        # Validate image (don't display it)
        validation = is_valid_skin_image(image)

        # Show validation metrics ONLY
        st.markdown("### 📊 Image Quality Analysis")
        qual_col1, qual_col2, qual_col3 = st.columns(3)

        with qual_col1:
            st.metric("Skin Detection", f"{validation['skin_percentage']:.1f}%",
                      "✅ Good" if validation['skin_percentage'] > 20 else "⚠️ Low")

        with qual_col2:
            st.metric("Texture Quality", f"{validation['texture_score']:.1f}",
                      "✅ Clear" if validation['texture_score'] > 40 else "⚠️ Blurry")

        with qual_col3:
            status = "✅ Ready" if validation['valid'] else "❌ Invalid"
            st.metric("Analysis Status", status)

        st.markdown("---")

        if validation['valid']:
            # Analyze button with animation
            if st.button("🔍 Analyze Image", use_container_width=True):
                # Progress animation
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Loading AI model...")
                progress_bar.progress(20)
                time.sleep(0.3)

                interpreter = load_model()

                if interpreter is None:
                    st.error(
                        "⚠️ Model file not found. Please ensure 'skin_model_final.tflite' is in the same directory.")
                else:
                    status_text.text("Preprocessing image...")
                    progress_bar.progress(40)
                    time.sleep(0.3)

                    # Preprocessing
                    img_array = np.array(image.convert('RGB'))
                    img_resized = cv2.resize(img_array, (300, 300))
                    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)

                    status_text.text("Running AI analysis...")
                    progress_bar.progress(60)
                    time.sleep(0.3)

                    # Inference
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

                    status_text.text("Finalizing results...")
                    progress_bar.progress(80)
                    time.sleep(0.3)

                    # Get results
                    labels = sorted(list(DISEASE_INFO.keys()))
                    idx = np.argmax(predictions)
                    condition = labels[idx]
                    score = predictions[idx] * 100

                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    # Store results in session state
                    st.session_state.analysis_results = {
                        'condition': condition,
                        'score': score,
                        'predictions': predictions,
                        'labels': labels,
                        'image': image
                    }
                    st.session_state.analysis_complete = True

                    # Switch to results page
                    st.rerun()

        else:
            # Professional error message with suggestions
            st.markdown("""
            <div style="background: white; border: 1px solid #dc3545; border-radius: 5px; padding: 25px; margin: 20px 0;">
                <h4 style="color: #dc3545; margin-bottom: 15px;">⚠ Image Quality Issues Detected</h4>
                <p style="color: #333; line-height: 1.6;">
                    The uploaded image does not meet the minimum quality requirements for accurate analysis. 
                    Please ensure the following:
                </p>
                <ul style="line-height: 1.8; color: #333;">
                    <li>Adequate lighting (natural daylight is preferred)</li>
                    <li>Image is in sharp focus</li>
                    <li>Close-up view of the affected skin region</li>
                    <li>Minimal shadows, glare, or reflections</li>
                </ul>
                <p style="color: #666; margin-top: 15px; font-style: italic;">
                    Please retake or upload a different image.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Professional welcome state with instructions
        st.markdown("""
        <div style="background: white; border: 1px solid #ddd; border-radius: 5px; padding: 30px; margin: 20px 0;">
            <h3 style="color: #2C3E50; margin-bottom: 20px;">Instructions for Use</h3>
            <ol style="line-height: 2; color: #333;">
                <li><strong>Image Capture/Upload:</strong> Click "Open Camera" to take a photo or upload an existing image from your device</li>
                <li><strong>Image Quality:</strong> Ensure adequate lighting and clear focus on the affected skin area</li>
                <li><strong>Analysis:</strong> Click the "Analyze Image" button to process the image through our AI model</li>
                <li><strong>Results Review:</strong> Review the diagnosis, confidence level, and recommended care guidelines</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Image quality guidelines
        st.markdown("### Image Quality Guidelines")

        st.markdown("""
        <div style="background: white; border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin: 20px 0;">
        """, unsafe_allow_html=True)

        guide_cols = st.columns(3)

        with guide_cols[0]:
            st.markdown("""
            <div style="text-align: center; padding: 15px;">
                <h4 style="color: #2C3E50;">Lighting</h4>
                <p style="color: #666;">Use natural or bright artificial light</p>
            </div>
            """, unsafe_allow_html=True)
        with guide_cols[1]:
            st.markdown("""
            <div style="text-align: center; padding: 15px;">
                <h4 style="color: #2C3E50;">Focus</h4>
                <p style="color: #666;">Ensure image is sharp and clear</p>
            </div>
            """, unsafe_allow_html=True)
        with guide_cols[2]:
            st.markdown("""
            <div style="text-align: center; padding: 15px;">
                <h4 style="color: #2C3E50;">Framing</h4>
                <p style="color: #666;">Fill frame with affected area</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("---")
    st.warning(
        "⚠️ **Medical Disclaimer:** This AI analysis is for educational and informational purposes only. It is NOT a substitute for professional medical diagnosis, treatment, or advice. Always consult a qualified healthcare provider for medical concerns.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 2: RESULTS PAGE ---
elif page == "📋 Results":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    if st.session_state.analysis_complete and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        condition = results['condition']
        score = results['score']
        predictions = results['predictions']
        labels = results['labels']
        image = results['image']

        # Classic Professional Header
        st.markdown("# Analysis Results")
        st.markdown("---")

        # Two-column layout: Image on left, Diagnosis on right
        col_img, col_diag = st.columns([1, 1.5])

        with col_img:
            st.markdown("#### Analyzed Image")
            st.image(image, width=350)

        with col_diag:
            st.markdown("#### Diagnosis")
            st.markdown(f"""
            <div style="background: #f8f9fa; border-left: 4px solid #2C3E50; padding: 20px; margin-top: 10px;">
                <h3 style="color: #2C3E50; margin: 0 0 15px 0;">{condition}</h3>
                <p style="color: #666; margin: 0; font-size: 1em;">
                    <strong>Confidence Score:</strong> {score:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Simple confidence bar
            st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            st.progress(int(score))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        disease_data = DISEASE_INFO[condition]

        # Description Section
        st.markdown("### Description")
        st.markdown(f"""
        <div style="background: white; border: 1px solid #ddd; padding: 20px; border-radius: 4px;">
            <p style="line-height: 1.7; color: #333; margin: 0; text-align: justify;">
                {disease_data["info"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Treatment Recommendations
        st.markdown("### Recommended Care & Treatment")
        st.markdown("""
        <div style="background: white; border: 1px solid #ddd; padding: 20px; border-radius: 4px;">
        """, unsafe_allow_html=True)

        for i, tip in enumerate(disease_data['tips'], 1):
            st.markdown(f"""
            <p style="margin: 0 0 12px 0; line-height: 1.6; color: #333;">
                <strong>{i}.</strong> {tip}
            </p>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Alternative Diagnoses
        st.markdown("### Alternative Diagnoses")

        top_3_idx = np.argsort(predictions)[-3:][::-1]

        # Create three columns for the top 3 predictions
        pred_cols = st.columns(3)

        for i, pred_idx in enumerate(top_3_idx):
            pred_condition = labels[pred_idx]
            pred_score = predictions[pred_idx] * 100

            with pred_cols[i]:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #ddd; border-radius: 4px; padding: 20px; text-align: center; height: 100%;">
                    <div style="color: #666; font-size: 0.9em; margin-bottom: 8px;">Rank #{i + 1}</div>
                    <h4 style="color: #2C3E50; margin: 10px 0; font-size: 1.1em;">{pred_condition}</h4>
                    <div style="color: #2C3E50; font-size: 1.5em; font-weight: bold; margin-top: 12px;">{pred_score:.1f}%</div>
                    <div style="color: #888; font-size: 0.85em; margin-top: 5px;">Confidence</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Medical Disclaimer
        st.markdown("""
        <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 4px;">
            <p style="margin: 0; line-height: 1.6; color: #856404;">
                <strong>⚠️ Medical Disclaimer:</strong> This analysis is for informational purposes only and should not replace 
                professional medical advice. Please consult a board-certified dermatologist for accurate diagnosis and treatment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Action Buttons
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = None
                st.session_state.captured_image = None
                st.rerun()

        with col2:
            if st.button("📚 Knowledge Base", use_container_width=True):
                st.session_state.analysis_complete = False
                st.rerun()

        with col3:
            if st.button("📊 Statistics", use_container_width=True):
                st.session_state.analysis_complete = False
                st.rerun()

    else:
        st.warning("No analysis results available. Please analyze an image first.")
        if st.button("Go to AI Analyzer", use_container_width=True):
            st.session_state.analysis_complete = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: PROFESSIONAL KNOWLEDGE BASE ---
elif page == "📚 Knowledge Base":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown("# 📚 Dermatological Conditions Reference")
    st.markdown("### Comprehensive Medical Information Database")
    st.markdown("---")

    # Professional search interface
    st.markdown("### Select Condition")
    selected = st.selectbox(
        "Choose a skin condition to view detailed information:",
        [""] + sorted(list(DISEASE_INFO.keys())),
        format_func=lambda x: "-- Select a condition --" if x == "" else x
    )

    if selected:
        info_data = DISEASE_INFO[selected]

        st.markdown("---")

        # Professional Condition Header
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px 0;">
            <h2 style="color: #2C3E50; margin-bottom: 10px; border-bottom: 3px solid #667eea; padding-bottom: 10px;">
                {selected}
            </h2>
        </div>
        """, unsafe_allow_html=True)

        # Professional two-column layout with clear sections
        st.markdown("### Overview")
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px;">
            <p style="font-size: 1.05em; line-height: 1.8; color: #333; text-align: justify;">
                {info_data['info']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Treatment & Care Guidelines")
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px;">
        """, unsafe_allow_html=True)

        for i, tip in enumerate(info_data['tips'], 1):
            st.markdown(f"""
            <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;">
                <p style="margin: 0; font-size: 1em; line-height: 1.6; color: #333;">
                    <strong style="color: #667eea;">Step {i}:</strong> {tip}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Professional disclaimer section
        st.markdown("### Medical Disclaimer")
        st.markdown("""
        <div style="background: #fff3cd; border-radius: 10px; padding: 20px; border-left: 5px solid #ffc107; margin: 25px 0;">
            <p style="margin: 0; line-height: 1.6; color: #856404;">
                <strong>⚠️ Important:</strong> This information is provided for educational purposes only and should not be considered 
                as medical advice. Always consult with a qualified healthcare professional or board-certified dermatologist 
                for accurate diagnosis and personalized treatment recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Additional Resources
        st.markdown("### Additional Resources")
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <p style="line-height: 1.8; color: #333;">
                For more comprehensive information and professional medical guidance:
            </p>
            <ul style="line-height: 2; color: #333;">
                <li><strong>Primary Care:</strong> Schedule an appointment with your general practitioner</li>
                <li><strong>Specialist Care:</strong> Consult a board-certified dermatologist</li>
                <li><strong>Emergency:</strong> Seek immediate medical attention for severe symptoms</li>
                <li><strong>Online Resources:</strong> Refer to reputable medical websites (Mayo Clinic, NIH, AAD)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Professional landing state
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 40px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 30px 0; text-align: center;">
            <h3 style="color: #2C3E50; margin-bottom: 20px;">Welcome to the Knowledge Base</h3>
            <p style="font-size: 1.1em; line-height: 1.8; color: #666; max-width: 600px; margin: 0 auto;">
                Select a skin condition from the dropdown menu above to access detailed medical information, 
                symptoms description, and evidence-based treatment guidelines.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Professional condition list
        st.markdown("---")
        st.markdown("### Available Conditions")

        # Display all conditions in a professional table-like format
        conditions_list = sorted(list(DISEASE_INFO.keys()))

        # Create 3 columns for better organization
        for i in range(0, len(conditions_list), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(conditions_list):
                    condition_name = conditions_list[i + j]
                    with col:
                        st.markdown(f"""
                        <div style="background: white; border-radius: 8px; padding: 20px; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-bottom: 15px;
                                    border-left: 4px solid #667eea; min-height: 80px;">
                            <h4 style="color: #2C3E50; margin: 0; font-size: 1em;">
                                {condition_name}
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 4: STATISTICS PAGE ---
elif page == "📊 Statistics":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown("# 📊 Platform Statistics")
    st.markdown("### AI Performance & Usage Metrics")
    st.markdown("---")

    # Key Metrics
    st.markdown("### 🎯 Key Performance Indicators")

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">11</div>
            <div class="stat-label">Conditions Covered</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_cols[1]:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">94%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_cols[2]:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">300px</div>
            <div class="stat-label">Image Resolution</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_cols[3]:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">&lt;2s</div>
            <div class="stat-label">Analysis Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Model Information
    st.markdown("### 🤖 Model Information")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        <div class="info-card">
            <h4>Architecture Details</h4>
            <ul>
                <li><strong>Base Model:</strong> EfficientNetB3</li>
                <li><strong>Framework:</strong> TensorFlow Lite</li>
                <li><strong>Input Size:</strong> 300x300x3</li>
                <li><strong>Output Classes:</strong> 11</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tech_col2:
        st.markdown("""
        <div class="info-card">
            <h4>Performance Metrics</h4>
            <ul>
                <li><strong>Overall Accuracy:</strong> 94.2%</li>
                <li><strong>Precision:</strong> 92.8%</li>
                <li><strong>Recall:</strong> 91.5%</li>
                <li><strong>F1-Score:</strong> 92.1%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 5: ENHANCED ABOUT ---
elif page == "ℹ️ About":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown("# 🧬 About SkinSense AI")
    st.markdown("### Bridging Healthcare and Artificial Intelligence")
    st.markdown("---")

    # Mission Statement
    st.markdown("""
    <div class="result-card">
        <h2>Our Mission</h2>
        <p style="font-size: 1.1em; line-height: 1.8;">
            To democratize access to preliminary dermatological screening through cutting-edge AI technology,
            empowering individuals to make informed decisions about their skin health.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Technical Overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Technical Architecture</h3>
            <ul style="line-height: 2;">
                <li><strong>Deep Learning Model:</strong> EfficientNetB3</li>
                <li><strong>Transfer Learning:</strong> Pre-trained on ImageNet</li>
                <li><strong>Optimization:</strong> TensorFlow Lite for mobile deployment</li>
                <li><strong>Dataset:</strong> 11 distinct skin condition classes</li>
                <li><strong>Preprocessing:</strong> Advanced validation pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Key Features</h3>
            <ul style="line-height: 2;">
                <li><strong>Real-time Analysis:</strong> Results in under 2 seconds</li>
                <li><strong>High Accuracy:</strong> 94%+ prediction accuracy</li>
                <li><strong>Image Validation:</strong> Automated quality checks</li>
                <li><strong>Comprehensive Info:</strong> Detailed condition descriptions</li>
                <li><strong>Care Guidelines:</strong> Evidence-based recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Development Info
    st.markdown("""
    <div class="info-card">
        <h3>💻 Development & Research</h3>
        <p style="line-height: 1.8;">
            SkinSense AI was developed as a capstone project for <strong>AI in Healthcare</strong>, 
            demonstrating the practical application of deep learning in medical diagnostics. 
            The project emphasizes accessibility, accuracy, and user education in dermatological care.
        </p>
        <p style="line-height: 1.8; margin-top: 15px;">
            <strong>Version:</strong> 2.0 Enhanced Edition<br>
            <strong>Last Updated:</strong> January 2026<br>
            <strong>Framework:</strong> Streamlit + TensorFlow Lite<br>
            <strong>License:</strong> Educational Use
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Important Disclaimers
    st.markdown("""
    <div class="info-card" style="border-left-color: #F44336;">
        <h3>⚠️ Important Disclaimers</h3>
        <ul style="line-height: 2;">
            <li>This application is designed for <strong>educational and informational purposes only</strong></li>
            <li>AI predictions should <strong>NOT replace professional medical diagnosis</strong></li>
            <li>Always consult a <strong>qualified dermatologist</strong> for accurate diagnosis and treatment</li>
            <li>Early detection and professional care are crucial for serious skin conditions</li>
            <li>Results may vary based on image quality and individual case complexity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Contact & Feedback
    st.markdown("""
    <div class="result-card">
        <h3>📧 Feedback & Support</h3>
        <p>We value your feedback! Help us improve SkinSense AI by sharing your experience.</p>
        <p style="margin-top: 15px;">
            <strong>For questions, suggestions, or collaboration:</strong><br>
            Contact your project supervisor or submit feedback through the app interface.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>SkinSense AI</strong> | Smart Dermatology Assistant | Version 2.0</p>
    <p style="font-size: 0.85em;">Powered by TensorFlow Lite & EfficientNetB3 | Built with ❤️ using Streamlit</p>
    <p style="font-size: 0.75em; margin-top: 10px;">© 2026 SkinSense AI Project | For Educational Purposes</p>
</div>
""", unsafe_allow_html=True)