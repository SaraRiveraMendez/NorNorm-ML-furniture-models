import os
import zipfile
import yaml
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import json


class FurnitureRecognitionModel:
    def __init__(self, img_size=(1024, 1024), batch_size=16):
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_names = []

    def extract_dataset(self, zip_path, extract_path="dataset/"):
        """Extract the YOLOv12 format dataset from zip file"""
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Dataset extracted to: {extract_path}")
        return extract_path

    def parse_yolo_annotations(self, dataset_path):
        """Parse YOLOv12 format annotations and convert to classification format"""
        print("Parsing YOLO annotations...")

        # Read data.yaml to get class names
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])

        # Process train and valid directories
        train_images = []
        train_labels = []
        val_images = []
        val_labels = []

        for split in ["train", "valid"]:
            images_dir = os.path.join(dataset_path, split, "images")
            labels_dir = os.path.join(dataset_path, split, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue

            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(images_dir, img_file)
                    label_file = os.path.splitext(img_file)[0] + ".txt"
                    label_path = os.path.join(labels_dir, label_file)

                    if os.path.exists(label_path):
                        # Read YOLO annotation (format: class_id center_x center_y width height)
                        with open(label_path, "r") as f:
                            lines = f.readlines()
                            if lines:
                                # Take the first object's class (you can modify this logic)
                                class_id = int(lines[0].split()[0])

                                if split == "train":
                                    train_images.append(img_path)
                                    train_labels.append(class_id)
                                else:
                                    val_images.append(img_path)
                                    val_labels.append(class_id)

        print(
            f"Found {len(train_images)} training images and {len(val_images)} validation images"
        )
        print(f"Classes: {self.class_names}")

        return (train_images, train_labels), (val_images, val_labels)

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image"""
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            image = image.astype("float32") / 255.0
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def create_data_generators(self, train_data, val_data):
        """Create data generators for training"""
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Load and preprocess images
        print("Loading training images...")
        X_train = []
        y_train = []
        for img_path, label in zip(train_images, train_labels):
            img = self.load_and_preprocess_image(img_path)
            if img is not None:
                X_train.append(img)
                y_train.append(label)

        print("Loading validation images...")
        X_val = []
        y_val = []
        for img_path, label in zip(val_images, val_labels):
            img = self.load_and_preprocess_image(img_path)
            if img is not None:
                X_val.append(img)
                y_val.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(
            y_train, num_classes=len(self.class_names)
        )
        y_val_cat = tf.keras.utils.to_categorical(
            y_val, num_classes=len(self.class_names)
        )

        # No data augmentation - use original images only
        train_datagen = ImageDataGenerator()
        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(
            X_train, y_train_cat, batch_size=self.batch_size, shuffle=True
        )
        val_generator = val_datagen.flow(
            X_val, y_val_cat, batch_size=self.batch_size, shuffle=False
        )

        return train_generator, val_generator

    def build_model(self, num_classes):
        """Build CNN model for furniture classification"""
        print("Building model...")

        # Using transfer learning with MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3), include_top=False, weights="imagenet"
        )
        base_model.trainable = False

        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def train_model(self, train_generator, val_generator, epochs=50):
        """Train the furniture recognition model"""
        print("Starting training...")

        # Callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                "Models/best_furniture_model.h5",
                monitor="val_accuracy",
                save_best_only=True,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
        ]

        # Calculate steps per epoch
        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)

        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def fine_tune_model(self, train_generator, val_generator, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        print("Starting fine-tuning...")

        # Unfreeze the top layers of the base model
        self.model.layers[0].trainable = True

        # Fine-tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before fine_tune_at
        for layer in self.model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False

        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001 / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                "Models/fine_tuned_furniture_model.h5",
                monitor="val_accuracy",
                save_best_only=True,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
        ]

        history_fine = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=callbacks,
            verbose=1,
        )

        return history_fine

    def plot_training_history(self, history, history_fine=None):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        if history_fine:
            total_epochs = len(history.history["accuracy"])
            plt.plot(
                range(
                    total_epochs, total_epochs + len(history_fine.history["accuracy"])
                ),
                history_fine.history["accuracy"],
                label="Fine-tuning Training Accuracy",
            )
            plt.plot(
                range(
                    total_epochs,
                    total_epochs + len(history_fine.history["val_accuracy"]),
                ),
                history_fine.history["val_accuracy"],
                label="Fine-tuning Validation Accuracy",
            )
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        if history_fine:
            total_epochs = len(history.history["loss"])
            plt.plot(
                range(total_epochs, total_epochs + len(history_fine.history["loss"])),
                history_fine.history["loss"],
                label="Fine-tuning Training Loss",
            )
            plt.plot(
                range(
                    total_epochs, total_epochs + len(history_fine.history["val_loss"])
                ),
                history_fine.history["val_loss"],
                label="Fine-tuning Validation Loss",
            )
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, image_path):
        """Predict furniture class for a single image"""
        if self.model is None:
            print("Model not trained yet!")
            return None

        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return None

        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return {
            "class": (
                self.class_names[predicted_class]
                if self.class_names
                else f"Class_{predicted_class}"
            ),
            "confidence": float(confidence),
            "class_id": int(predicted_class),
        }

    def save_model(self, filepath="Models/furniture_recognition_model.h5"):
        """Save the trained model"""
        if self.model:
            # Create Models directory if it doesn't exist
            os.makedirs("Models", exist_ok=True)
            self.model.save(filepath)
            # Save class names
            with open(filepath.replace(".h5", "_classes.json"), "w") as f:
                json.dump(self.class_names, f)
            print(f"Model saved to: {filepath}")

    def load_model(self, filepath="Models/furniture_recognition_model.h5"):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        # Load class names
        try:
            with open(filepath.replace(".h5", "_classes.json"), "r") as f:
                self.class_names = json.load(f)
        except FileNotFoundError:
            print("Class names file not found. Please set class_names manually.")
        print(f"Model loaded from: {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize the furniture recognition model
    furniture_model = FurnitureRecognitionModel(img_size=(1024, 1024), batch_size=16)

    # Step 1: Extract dataset from zip
    dataset_path = furniture_model.extract_dataset(
        "data/The-Mexican-Project.v1-zoom-rotation-dataset.yolov12.zip"
    )

    # Step 2: Parse YOLO annotations
    train_data, val_data = furniture_model.parse_yolo_annotations(dataset_path)

    # Step 3: Create data generators
    train_generator, val_generator = furniture_model.create_data_generators(
        train_data, val_data
    )

    # Step 4: Build model
    model = furniture_model.build_model(num_classes=len(furniture_model.class_names))
    print(model.summary())

    # Step 5: Train model
    history = furniture_model.train_model(train_generator, val_generator, epochs=50)

    # Step 6: Fine-tune model (optional)
    history_fine = furniture_model.fine_tune_model(
        train_generator, val_generator, epochs=20
    )

    # Step 7: Plot training history
    furniture_model.plot_training_history(history, history_fine)

    # Step 8: Save model
    furniture_model.save_model("Models/furniture_recognition_model.h5")
