import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# FACE MASK DETECTION - CNN TRAINING
# ============================================================

# -----------------------------
# SETTINGS
# -----------------------------

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VALID_DIR = os.path.join(BASE_DIR, "dataset", "valid")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "mask_detector.keras")


# ============================================================
# CHECK DIRECTORIES
# ============================================================

def check_directories():

    required_dirs = [
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR
    ]

    for directory in required_dirs:

        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"\nDataset folder not found:\n{directory}\n"
            )

        csv_file = os.path.join(directory, "annotations.csv")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"\nannotations.csv not found:\n{csv_file}\n"
            )

    os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(folder):

    csv_path = os.path.join(folder, "annotations.csv")

    print("\n--------------------------------------------")
    print(f"Loading dataset: {folder}")
    print("--------------------------------------------")

    # Read annotations
    dataframe = pd.read_csv(csv_path)

    # Required columns
    required_columns = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax"
    ]

    # Check CSV columns
    for column in required_columns:

        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' is missing from {csv_path}"
            )

    images = []
    labels = []

    total_rows = len(dataframe)

    print(f"Annotations found: {total_rows}")

    # --------------------------------------------------------
    # Process every annotation
    # --------------------------------------------------------

    for index, row in dataframe.iterrows():

        try:

            # -----------------------------
            # Image filename
            # -----------------------------

            filename = str(row["filename"]).strip()

            image_path = os.path.join(folder, filename)

            if not os.path.isfile(image_path):

                print(
                    f"Warning: Image not found: {filename}"
                )

                continue

            # -----------------------------
            # Read image
            # -----------------------------

            image = cv2.imread(image_path)

            if image is None:

                print(
                    f"Warning: Could not read image: {filename}"
                )

                continue

            # -----------------------------
            # Bounding box
            # -----------------------------

            xmin = int(float(row["xmin"]))
            ymin = int(float(row["ymin"]))
            xmax = int(float(row["xmax"]))
            ymax = int(float(row["ymax"]))

            # Image dimensions
            height, width = image.shape[:2]

            # Keep coordinates inside image
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))

            xmax = max(0, min(xmax, width))
            ymax = max(0, min(ymax, height))

            # Check bounding box
            if xmax <= xmin or ymax <= ymin:

                print(
                    f"Warning: Invalid bounding box: {filename}"
                )

                continue

            # -----------------------------
            # Crop face
            # -----------------------------

            face = image[ymin:ymax, xmin:xmax]

            if face.size == 0:

                print(
                    f"Warning: Empty face crop: {filename}"
                )

                continue

            # -----------------------------
            # Resize
            # -----------------------------

            face = cv2.resize(
                face,
                (IMG_SIZE, IMG_SIZE)
            )

            # -----------------------------
            # BGR → RGB
            # -----------------------------

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # -----------------------------
            # Normalize
            # -----------------------------

            face = face.astype(
                np.float32
            ) / 255.0

            # -----------------------------
            # Class label
            # -----------------------------

            class_name = str(
                row["class"]
            ).strip().lower()

            if class_name == "mask":

                label = 0

            elif class_name in [
                "no-mask",
                "no_mask",
                "no mask",
                "nomask"
            ]:

                label = 1

            else:

                print(
                    f"Warning: Unknown class '{class_name}' "
                    f"in {filename}"
                )

                continue

            # -----------------------------
            # Store image and label
            # -----------------------------

            images.append(face)
            labels.append(label)

        except Exception as error:

            print(
                f"Warning processing row {index}: {error}"
            )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    if len(images) == 0:

        raise ValueError(
            f"\nNo valid images were loaded from:\n{folder}\n"
        )

    images = np.array(
        images,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.float32
    )

    print(f"Images loaded : {len(images)}")
    print(f"Labels loaded : {len(labels)}")

    # Class distribution
    mask_count = np.sum(labels == 0)
    no_mask_count = np.sum(labels == 1)

    print(f"Mask images   : {mask_count}")
    print(f"No-mask images: {no_mask_count}")

    return images, labels


# ============================================================
# BUILD CNN MODEL
# ============================================================

def create_model():

    model = models.Sequential([

        # Input
        layers.Input(
            shape=(IMG_SIZE, IMG_SIZE, 3)
        ),

        # -----------------------------
        # Convolution Block 1
        # -----------------------------

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # -----------------------------
        # Convolution Block 2
        # -----------------------------

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # -----------------------------
        # Convolution Block 3
        # -----------------------------

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # -----------------------------
        # Regularization
        # -----------------------------

        layers.Dropout(0.3),

        # -----------------------------
        # Flatten
        # -----------------------------

        layers.Flatten(),

        # -----------------------------
        # Dense Layer
        # -----------------------------

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        # -----------------------------
        # Output
        # 0 = Mask
        # 1 = No Mask
        # -----------------------------

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("==============================================")
    print("       FACE MASK DETECTION - CNN")
    print("==============================================")

    # -----------------------------
    # Check folders
    # -----------------------------

    print("\nChecking dataset folders...")

    check_directories()

    print("Dataset folders found successfully.")

    # -----------------------------
    # Load training data
    # -----------------------------

    X_train, y_train = load_dataset(
        TRAIN_DIR
    )

    # -----------------------------
    # Load validation data
    # -----------------------------

    X_valid, y_valid = load_dataset(
        VALID_DIR
    )

    # -----------------------------
    # Load test data
    # -----------------------------

    X_test, y_test = load_dataset(
        TEST_DIR
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n")
    print("==============================================")
    print("              DATASET SUMMARY")
    print("==============================================")

    print(
        f"Training images   : {len(X_train)}"
    )

    print(
        f"Validation images : {len(X_valid)}"
    )

    print(
        f"Testing images    : {len(X_test)}"
    )

    print(
        f"Image size        : {IMG_SIZE} x {IMG_SIZE}"
    )

    print("==============================================")

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\nCreating CNN model...")

    model = create_model()

    # Display model
    model.summary()

    # ========================================================
    # CALLBACKS
    # ========================================================

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print("\n")
    print("==============================================")
    print("            STARTING TRAINING")
    print("==============================================")

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_valid,
            y_valid
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping,
            checkpoint
        ],

        shuffle=True,

        verbose=1
    )

    # ========================================================
    # TEST MODEL
    # ========================================================

    print("\n")
    print("==============================================")
    print("              TESTING MODEL")
    print("==============================================")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\n")
    print(
        f"Test Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy * 100:.2f}%"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(MODEL_PATH)

    print("\n")
    print("==============================================")
    print("          TRAINING COMPLETED")
    print("==============================================")

    print(
        f"Model saved successfully:"
    )

    print(
        MODEL_PATH
    )

    print("==============================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()