import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# --------------------------------
# Custom CSS
# --------------------------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result {
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Title
# --------------------------------

st.markdown(
    '<div class="main-title">😷 Face Mask Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CNN + TensorFlow + OpenCV</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload an image and the AI model will detect whether "
    "a person is wearing a face mask."
)

# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/mask_detector.keras"
    )

model = load_model()

# --------------------------------
# Load Face Detector
# --------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------------
# Image Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Process Image
# --------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    result_image = image_array.copy()

    # --------------------------------
    # No face detected
    # --------------------------------

    if len(faces) == 0:

        st.warning(
            "⚠️ No face detected. Please upload an image containing a clear face."
        )

    # --------------------------------
    # Process detected faces
    # --------------------------------

    else:

        st.success(
            f"✅ {len(faces)} face(s) detected"
        )

        for (x, y, w, h) in faces:

            face = image_array[
                y:y+h,
                x:x+w
            ]

            if face.size == 0:
                continue

            # Resize
            face = cv2.resize(
                face,
                (128, 128)
            )

            # Normalize
            face = face.astype(
                "float32"
            ) / 255.0

            # Add batch dimension
            face = np.expand_dims(
                face,
                axis=0
            )

            # Prediction
            prediction = model.predict(
                face,
                verbose=0
            )[0][0]

            # Classification
            if prediction >= 0.5:

                label = "NO MASK"

                confidence = (
                    prediction * 100
                )

                box_color = (255, 0, 0)

            else:

                label = "MASK"

                confidence = (
                    (1 - prediction) * 100
                )

                box_color = (0, 200, 0)

            # Draw rectangle
            cv2.rectangle(
                result_image,
                (x, y),
                (x+w, y+h),
                box_color,
                3
            )

            # Label
            text = (
                f"{label}: "
                f"{confidence:.1f}%"
            )

            cv2.putText(
                result_image,
                text,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

            # Display result
            st.markdown(
                f"""
                <div class="result">
                {label}<br>
                Confidence: {confidence:.1f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # --------------------------------
        # Display processed image
        # --------------------------------

        st.subheader("🔍 Detection Result")

        st.image(
            result_image,
            caption="Face Mask Detection Result",
            use_container_width=True
        )

# --------------------------------
# Footer
# --------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
    <b>Face Mask Detection</b><br>
    Deep Learning + Computer Vision Project
    </div>
    """,
    unsafe_allow_html=True
)