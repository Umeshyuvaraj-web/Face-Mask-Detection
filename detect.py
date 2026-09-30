import cv2
import tensorflow as tf
import numpy as np
import time

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "model/mask_detector.keras"
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# -----------------------------
# Face detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

IMG_SIZE = 128

# FPS calculation
previous_time = 0

print("Webcam started!")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Flip image horizontally for natural webcam view
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # -----------------------------
    # Detect mask for each face
    # -----------------------------
    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # Resize
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        # Convert BGR → RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize
        face = face.astype("float32") / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Prediction
        prediction = model.predict(face, verbose=0)[0][0]

        # -----------------------------
        # Classification
        # -----------------------------
        if prediction >= 0.5:
            label = "NO MASK"
            confidence = prediction * 100
            box_color = (0, 0, 255)      # Red
        else:
            label = "MASK"
            confidence = (1 - prediction) * 100
            box_color = (0, 255, 0)      # Green

        # -----------------------------
        # Draw face rectangle
        # -----------------------------
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            box_color,
            3
        )

        # Label text
        text = f"{label}: {confidence:.1f}%"

        # Background for text
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        cv2.rectangle(
            frame,
            (x, y - text_height - 15),
            (x + text_width + 10, y),
            box_color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # -----------------------------
    # Calculate FPS
    # -----------------------------
    current_time = time.time()

    fps = 1 / (current_time - previous_time) if previous_time != 0 else 0

    previous_time = current_time

    # FPS display
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Project title
    cv2.putText(
        frame,
        "AI FACE MASK DETECTION",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Show webcam
    cv2.imshow("Face Mask Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Release resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("Detection stopped.")