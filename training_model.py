import json
import os
import zipfile
from pathlib import Path

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


class OptimizedFurnitureModel:
    def __init__(self, img_size=(1024, 1024), batch_size=8):
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_names = []

    def download_and_extract_remote_dataset(self, gdrive_file_id, output_filename=None):
        """Download dataset from Google Drive and extract without storing zip permanently"""
        import tempfile

        if output_filename is None:
            output_filename = f"dataset_{gdrive_file_id}.zip"

        url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        # Use temporary directory to avoid storage overhead
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, output_filename)

            print("Downloading dataset from Google Drive...")
            gdown.download(url, temp_zip_path, quiet=False)

            # Extract to permanent location
            extract_path = "dataset/"
            print("Extracting dataset...")
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # temp_zip_path is automatically deleted when temp_dir context exits

        return extract_path

    def stream_remote_dataset(self, gdrive_file_id, chunk_size=8192):
        """Stream and process dataset without storing locally"""
        import tempfile
        from io import BytesIO

        import requests

        # Direct download URL for Google Drive
        url = f"https://drive.google.com/uc?export=download&id={gdrive_file_id}"

        print("Streaming dataset from remote source...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Stream download
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Process zip in memory
            zip_content = BytesIO()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    zip_content.write(chunk)

            zip_content.seek(0)

            # Extract to temporary location
            extract_path = os.path.join(temp_dir, "dataset")
            with zipfile.ZipFile(zip_content, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Parse annotations immediately
            train_data, val_data = self.parse_yolo_annotations(extract_path)

        return train_data, val_data

    def parse_yolo_annotations(self, dataset_path):
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])

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
                        with open(label_path, "r") as f:
                            lines = f.readlines()
                            if lines:
                                class_id = int(lines[0].split()[0])
                                if split == "train":
                                    train_images.append(img_path)
                                    train_labels.append(class_id)
                                else:
                                    val_images.append(img_path)
                                    val_labels.append(class_id)

        print(f"Training: {len(train_images)}, Validation: {len(val_images)}")
        return (train_images, train_labels), (val_images, val_labels)

    def load_and_preprocess_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            image = image.astype("float32") / 255.0
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def create_memory_efficient_generators(self, train_data, val_data):
        """Create memory-efficient data generators that load images on-demand"""
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Convert labels to categorical format
        y_train_cat = tf.keras.utils.to_categorical(train_labels, num_classes=len(self.class_names))
        y_val_cat = tf.keras.utils.to_categorical(val_labels, num_classes=len(self.class_names))

        # Custom generator class for memory efficiency
        class MemoryEfficientSequence(tf.keras.utils.Sequence):
            def __init__(self, image_paths, labels, batch_size, img_size, shuffle=True):
                self.image_paths = image_paths
                self.labels = labels
                self.batch_size = batch_size
                self.img_size = img_size
                self.shuffle = shuffle
                self.indices = np.arange(len(self.image_paths))
                self.on_epoch_end()

            def __len__(self):
                return len(self.image_paths) // self.batch_size

            def __getitem__(self, index):
                batch_indices = self.indices[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                batch_paths = [self.image_paths[i] for i in batch_indices]
                batch_labels = self.labels[batch_indices]

                # Load images on-demand
                batch_images = []
                valid_labels = []

                for img_path, label in zip(batch_paths, batch_labels):
                    try:
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, self.img_size)
                        image = image.astype("float32") / 255.0
                        batch_images.append(image)
                        valid_labels.append(label)
                    except Exception:
                        continue

                if batch_images:
                    return np.array(batch_images), np.array(valid_labels)
                else:
                    # Return empty batch if all images failed to load
                    empty_batch = np.zeros((1, *self.img_size, 3))
                    empty_labels = np.zeros((1, len(self.labels[0])))
                    return empty_batch, empty_labels

            def on_epoch_end(self):
                if self.shuffle:
                    np.random.shuffle(self.indices)

        train_generator = MemoryEfficientSequence(
            train_images, y_train_cat, self.batch_size, self.img_size, shuffle=True
        )
        val_generator = MemoryEfficientSequence(
            val_images, y_val_cat, self.batch_size, self.img_size, shuffle=False
        )

        return train_generator, val_generator

    def build_model(self, num_classes):
        """Optimized architecture for small datasets with high regularization"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3), include_top=False, weights="imagenet"
        )

        # More aggressive layer freezing for small dataset
        base_model.trainable = True
        for layer in base_model.layers[:-20]:  # Only unfreeze last 20 layers
            layer.trainable = False

        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.8),  # Very high dropout for small dataset
                layers.Dense(
                    128, activation="relu", kernel_regularizer=l2(0.03)
                ),  # Strong L2 regularization
                layers.BatchNormalization(),
                layers.Dropout(0.7),
                layers.Dense(64, activation="relu", kernel_regularizer=l2(0.03)),
                layers.BatchNormalization(),
                layers.Dropout(0.6),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Very low learning rate for stability with small dataset
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def train_model(self, train_generator, val_generator, epochs=200):
        """Training with extended patience and conservative approach"""
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=30,  # Extended patience
                restore_best_weights=True,
                min_delta=0.001,
            ),
            ModelCheckpoint(
                "Models/best_furniture_model.h5", monitor="val_accuracy", save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,  # More aggressive reduction
                patience=15,
                min_lr=1e-8,
                cooldown=5,
            ),
        ]

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

    def evaluate_model_performance(self, history):
        """Analyze training performance and overfitting"""
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        best_val_acc = max(history.history["val_accuracy"])

        gap = final_train_acc - final_val_acc

        print(f"\nTraining Performance Summary:")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Overfitting Gap: {gap:.4f}")

        if gap > 0.3:
            print("Status: Severe overfitting detected")
            print("Recommendation: Increase regularization or collect more data")
        elif gap > 0.15:
            print("Status: Moderate overfitting")
            print("Recommendation: Consider additional regularization")
        else:
            print("Status: Good generalization")

    def plot_training_history(self, history):
        """Comprehensive training visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(history.history["accuracy"], label="Training", linewidth=2)
        axes[0, 0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(history.history["loss"], label="Training", linewidth=2)
        axes[0, 1].plot(history.history["val_loss"], label="Validation", linewidth=2)
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning Rate
        if "lr" in history.history:
            axes[1, 0].plot(history.history["lr"], linewidth=2)
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Overfitting analysis
        train_acc = np.array(history.history["accuracy"])
        val_acc = np.array(history.history["val_accuracy"])
        gap = train_acc - val_acc

        axes[1, 1].plot(gap, linewidth=2, color="red")
        axes[1, 1].set_title("Overfitting Gap (Train - Val Accuracy)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy Gap")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(
            y=0.1, color="orange", linestyle="--", alpha=0.7, label="Mild Overfitting"
        )
        axes[1, 1].axhline(
            y=0.2, color="red", linestyle="--", alpha=0.7, label="Severe Overfitting"
        )
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def predict_with_confidence_analysis(self, image_path, threshold=0.7):
        """Enhanced prediction with confidence analysis"""
        if self.model is None:
            return None

        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return None

        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image, verbose=0)

        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [(i, predictions[0][i]) for i in top_3_indices]

        result = {
            "predicted_class": (
                self.class_names[predicted_class]
                if self.class_names
                else f"Class_{predicted_class}"
            ),
            "confidence": float(confidence),
            "class_id": int(predicted_class),
            "is_confident": confidence > threshold,
            "top_3_predictions": [
                {
                    "class": self.class_names[i] if self.class_names else f"Class_{i}",
                    "confidence": float(conf),
                    "class_id": int(i),
                }
                for i, conf in top_3_predictions
            ],
        }

        return result

    def predict(self, image_path):
        """Standard prediction method for compatibility"""
        result = self.predict_with_confidence_analysis(image_path)
        if result:
            return {
                "class": result["predicted_class"],
                "confidence": result["confidence"],
                "class_id": result["class_id"],
            }
        return None

    def save_model(self, filepath="Models/optimized_furniture_model.h5"):
        if self.model:
            os.makedirs("Models", exist_ok=True)
            self.model.save(filepath)
            with open(filepath.replace(".h5", "_classes.json"), "w") as f:
                json.dump(self.class_names, f)

    def load_model(self, filepath="Models/optimized_furniture_model.h5"):
        self.model = tf.keras.models.load_model(filepath)
        try:
            with open(filepath.replace(".h5", "_classes.json"), "r") as f:
                self.class_names = json.load(f)
        except FileNotFoundError:
            print("Class names file not found")


if __name__ == "__main__":
    furniture_model = OptimizedFurnitureModel(img_size=(624, 624), batch_size=4)

    # Remote download with temporary storage
    gdrive_file_id = "1GvXf7jalRQw0Vd_17Dxa-vwlgPKJ7zCf"
    dataset_path = furniture_model.download_and_extract_remote_dataset(gdrive_file_id)
    train_data, val_data = furniture_model.parse_yolo_annotations(dataset_path)

    # Use memory-efficient generators
    train_generator, val_generator = furniture_model.create_memory_efficient_generators(
        train_data, val_data
    )

    model = furniture_model.build_model(num_classes=len(furniture_model.class_names))
    print(f"Model parameters: {model.count_params():,}")

    history = furniture_model.train_model(train_generator, val_generator, epochs=150)

    furniture_model.plot_training_history(history)
    furniture_model.evaluate_model_performance(history)

    furniture_model.save_model("Models/optimized_furniture_model.h5")

    # Clean up dataset folder to save space
    import shutil

    if os.path.exists("dataset/"):
        shutil.rmtree("dataset/")
        print("Dataset folder cleaned up to save storage space")
