"""
FlameVision - Fire Detection CNN
Binary classification model: FIRE vs NO FIRE
"""

import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 128
EPOCHS      = 10
BASE_PATH   = "dataset/FlameVision/archive/Classification"

TRAIN_PATH  = os.path.join(BASE_PATH, "train")
VAL_PATH    = os.path.join(BASE_PATH, "valid")
TEST_PATH   = os.path.join(BASE_PATH, "test")


# ─────────────────────────────────────────────
# DATA LOADING (Google Colab)
# ─────────────────────────────────────────────
def load_dataset_from_drive(zip_path: str) -> None:
    """Mount Google Drive and extract dataset ZIP."""
    from google.colab import drive
    drive.mount("/content/drive")

    print(f"Extracting dataset from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("dataset")
    print("Dataset extracted successfully.")


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(input_shape: tuple = (128, 128, 3)) -> tf.keras.Model:
    """
    4-block CNN for binary fire classification.

    Architecture:
        Conv2D(32) → Conv2D(64) → Conv2D(128) → Conv2D(128)
        GlobalAveragePooling → Dense(64) → Dense(1, sigmoid)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Classifier head
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ], name="FlameVision_CNN")

    return model


# ─────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────
def get_data_generators(
    train_path: str,
    val_path: str,
    test_path: str,
    img_size: tuple,
    batch_size: int,
) -> tuple:
    """Create ImageDataGenerators with augmentation for training."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(target_size=img_size, batch_size=batch_size, class_mode="binary")

    train_gen = train_datagen.flow_from_directory(train_path, shuffle=True,  seed=42, **common)
    val_gen   = val_datagen.flow_from_directory(val_path,   shuffle=False, **common)
    test_gen  = val_datagen.flow_from_directory(test_path,  shuffle=False, **common)

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def get_callbacks(checkpoint_path: str = "best_model.keras") -> list:
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train(model, train_gen, val_gen, epochs, callbacks) -> tuple:
    start = time.time()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start
    return history, elapsed


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, test_gen) -> dict:
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

    y_pred = (model.predict(test_gen, verbose=0) > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    print("\n── Test Results ──────────────────────────────")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_precision:.4f}")
    print(f"  Recall    : {test_recall:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")

    print("\n── Classification Report ─────────────────────")
    print(classification_report(y_true, y_pred, target_names=["FIRE", "NOFIRE"], digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print("── Confusion Matrix ──────────────────────────")
    print(f"               pred FIRE  pred NOFIRE")
    print(f"  actual FIRE   {cm[0,0]:6d}       {cm[0,1]:6d}")
    print(f"  actual NOFIRE {cm[1,0]:6d}       {cm[1,1]:6d}")

    return {"accuracy": test_acc, "precision": test_precision, "recall": test_recall, "f1": test_f1}


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────
def predict_image(image_path: str, model: tf.keras.Model, img_size: tuple = IMG_SIZE) -> tuple:
    """Predict a single image. Returns (label, confidence)."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    confidence = model.predict(arr, verbose=0)[0][0]
    label = "FIRE" if confidence > 0.5 else "NOFIRE"
    return label, float(confidence)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Uncomment the line below when running on Google Colab:
    # load_dataset_from_drive("/content/drive/MyDrive/datasets/FlameVision.zip")

    print("── Configuration ─────────────────────────────")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Max epochs : {EPOCHS}")

    # Model
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    # Data
    train_gen, val_gen, test_gen = get_data_generators(
        TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SIZE, BATCH_SIZE
    )
    print(f"\n  Classes      : {train_gen.class_indices}")
    print(f"  Train samples: {train_gen.samples}")
    print(f"  Val samples  : {val_gen.samples}")
    print(f"  Test samples : {test_gen.samples}")

    # Training
    history, elapsed = train(model, train_gen, val_gen, EPOCHS, get_callbacks())
    print(f"\n  Training time       : {elapsed:.1f}s")
    print(f"  Best val accuracy   : {max(history.history['val_accuracy']):.4f}")
    print(f"  Epochs completed    : {len(history.history['loss'])}")

    # Evaluation
    metrics = evaluate(model, test_gen)

    # Save model
    if metrics["accuracy"] > 0.85:
        model.save("fire_detection_model.keras")
        print(f"\n  Model saved → fire_detection_model.keras (acc: {metrics['accuracy']:.2%})")
    else:
        print(f"\n  Accuracy below threshold ({metrics['accuracy']:.2%}). Model not saved.")

    # Quick sanity check on a few test images
    print("\n── Sample Predictions ────────────────────────")
    for class_name in ["fire", "nofire"]:
        class_path = os.path.join(TEST_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if images:
            img_path = os.path.join(class_path, images[0])
            label, conf = predict_image(img_path, model)
            status = "✓" if label.lower() == class_name else "✗"
            print(f"  {status} {images[0]} — true: {class_name.upper()}, pred: {label} ({conf:.2%})")


if __name__ == "__main__":
    main()
