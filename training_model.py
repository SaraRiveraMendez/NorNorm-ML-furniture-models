import json
import os

# Clean up temporary dataset
import shutil
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


class ImprovedFurnitureModel:
    def __init__(self, img_size=(224, 224), batch_size=16):
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

        return extract_path

    def parse_yolo_annotations(self, dataset_path):
        """Extract ALL elements from each image, creating multiple training samples per image"""
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])
                print(f"Classes found: {self.class_names}")

        train_images = []
        train_labels = []
        val_images = []
        val_labels = []

        total_elements = 0
        images_processed = 0

        for split in ["train", "valid"]:
            images_dir = os.path.join(dataset_path, split, "images")
            labels_dir = os.path.join(dataset_path, split, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print(f"Directory {split} not found")
                continue

            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(images_dir, img_file)
                    label_file = os.path.splitext(img_file)[0] + ".txt"
                    label_path = os.path.join(labels_dir, label_file)

                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            lines = f.readlines()
                            elements_in_image = 0

                            # Process ALL lines (all elements in the image)
                            for line in lines:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:  # class_id, x, y, w, h
                                        class_id = int(parts[0])

                                        # Verify valid class_id
                                        if 0 <= class_id < len(self.class_names):
                                            if split == "train":
                                                train_images.append(img_path)
                                                train_labels.append(class_id)
                                            else:
                                                val_images.append(img_path)
                                                val_labels.append(class_id)

                                            elements_in_image += 1
                                            total_elements += 1

                            if elements_in_image > 0:
                                images_processed += 1

        print(f"Images processed: {images_processed}")
        print(f"Total elements extracted: {total_elements}")
        print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
        print(f"Average elements per image: {total_elements/images_processed:.2f}")

        # Show class distribution
        print(f"Class distribution in training:")
        unique, counts = np.unique(train_labels, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"Unknown_{class_id}"
            )
            print(f"  {class_name}: {count} samples")

        return (train_images, train_labels), (val_images, val_labels)

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            image = image.astype("float32") / 255.0
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def create_data_generators(self, train_data, val_data):
        """Create data generators without augmentation since dataset already includes it"""
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Encode labels to categorical
        num_classes = len(self.class_names)
        y_train_cat = tf.keras.utils.to_categorical(train_labels, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(val_labels, num_classes)

        class ImprovedSequence(tf.keras.utils.Sequence):
            def __init__(self, image_paths, labels, batch_size, img_size, shuffle=True):
                self.image_paths = image_paths
                self.labels = labels
                self.batch_size = batch_size
                self.img_size = img_size
                self.shuffle = shuffle
                self.indices = np.arange(len(self.image_paths))
                self.on_epoch_end()

            def __len__(self):
                return max(1, len(self.image_paths) // self.batch_size)

            def __getitem__(self, index):
                batch_indices = self.indices[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]

                batch_images = []
                batch_labels = []

                for idx in batch_indices:
                    if idx < len(self.image_paths):
                        img_path = self.image_paths[idx]
                        label = self.labels[idx]

                        image = self.load_and_preprocess_image_batch(img_path)
                        if image is not None:
                            batch_images.append(image)
                            batch_labels.append(label)

                # Ensure we have at least one image
                if not batch_images:
                    # Create valid empty batch
                    batch_images = [np.zeros((*self.img_size, 3))]
                    batch_labels = [
                        np.zeros(len(self.labels[0]) if len(self.labels) > 0 else num_classes)
                    ]

                return np.array(batch_images), np.array(batch_labels)

            def load_and_preprocess_image_batch(self, image_path):
                """Load and preprocess image for batch processing"""
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        return None
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.img_size)
                    image = image.astype("float32") / 255.0
                    return image
                except Exception:
                    return None

            def on_epoch_end(self):
                if self.shuffle:
                    np.random.shuffle(self.indices)

        train_generator = ImprovedSequence(
            train_images, y_train_cat, self.batch_size, self.img_size, shuffle=True
        )
        val_generator = ImprovedSequence(
            val_images, y_val_cat, self.batch_size, self.img_size, shuffle=False
        )

        return train_generator, val_generator

    def build_model(self, num_classes):
        """Build improved model with EfficientNetB0 and fine-tuning"""
        # Use EfficientNetB0 which is more modern and efficient
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*self.img_size, 3), include_top=False, weights="imagenet"
        )

        # Fine-tuning strategy
        base_model.trainable = True

        # Freeze initial layers, unfreeze last layers
        fine_tune_at = len(base_model.layers) - 30
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Build complete model architecture
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Set learning rate
        initial_learning_rate = 1e-4

        # Create top-3 metric if available
        try:
            top3_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
            metrics = ["accuracy", top3_metric]
        except AttributeError:
            metrics = ["accuracy"]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        self.model = model
        return model

    def train_model(self, train_generator, val_generator, epochs=80):
        """Train model with single fine-tuning approach"""
        print("Starting model training...")

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy", patience=15, restore_best_weights=True, min_delta=0.001
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8, min_lr=1e-8, verbose=1),
        ]

        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def evaluate_model_performance(self, history):
        """Analyze model performance in detail"""
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        best_val_acc = max(history.history["val_accuracy"])

        # Top-3 accuracy if available
        final_top3_acc = history.history.get("val_top_3_accuracy", [0])[-1]

        gap = final_train_acc - final_val_acc

        print(f"\n{'='*50}")
        print(f"MODEL PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"Final Training Accuracy:    {final_train_acc:.4f}")
        print(f"Final Validation Accuracy:  {final_val_acc:.4f}")
        print(f"Best Validation Accuracy:   {best_val_acc:.4f}")
        print(f"Top-3 Validation Accuracy:  {final_top3_acc:.4f}")
        print(f"Overfitting Gap:            {gap:.4f}")
        print(f"{'='*50}")

        if gap > 0.2:
            print("STATUS: Severe overfitting detected")
            print("RECOMMENDATION: Increase regularization or add more data")
        elif gap > 0.1:
            print("STATUS: Moderate overfitting")
            print("RECOMMENDATION: Consider additional regularization")
        else:
            print("STATUS: Good generalization")

        print(f"{'='*50}")

    def plot_training_history(self, history):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Complete Training Analysis", fontsize=16, fontweight="bold")

        # Accuracy plot
        axes[0, 0].plot(history.history["accuracy"], label="Training", linewidth=2, color="blue")
        axes[0, 0].plot(
            history.history["val_accuracy"], label="Validation", linewidth=2, color="red"
        )
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss plot
        axes[0, 1].plot(history.history["loss"], label="Training", linewidth=2, color="blue")
        axes[0, 1].plot(history.history["val_loss"], label="Validation", linewidth=2, color="red")
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Top-3 accuracy if available
        if "val_top_3_accuracy" in history.history:
            axes[0, 2].plot(
                history.history["top_3_accuracy"],
                label="Top-3 Training",
                linewidth=2,
                color="green",
            )
            axes[0, 2].plot(
                history.history["val_top_3_accuracy"],
                label="Top-3 Validation",
                linewidth=2,
                color="orange",
            )
            axes[0, 2].set_title("Top-3 Accuracy")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("Top-3 Accuracy")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # Learning rate plot
        if "lr" in history.history:
            axes[1, 0].plot(history.history["lr"], linewidth=2, color="purple")
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
        axes[1, 1].set_title("Overfitting Gap (Train - Val)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy Difference")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.05, color="yellow", linestyle="--", alpha=0.7, label="Ideal Limit")
        axes[1, 1].axhline(
            y=0.1, color="orange", linestyle="--", alpha=0.7, label="Mild Overfitting"
        )
        axes[1, 1].axhline(
            y=0.2, color="red", linestyle="--", alpha=0.7, label="Severe Overfitting"
        )
        axes[1, 1].legend()

        # Smoothed validation accuracy
        window = min(10, len(history.history["val_accuracy"]) // 4)
        if window > 1:
            val_acc_smooth = np.convolve(
                history.history["val_accuracy"], np.ones(window) / window, mode="valid"
            )
            axes[1, 2].plot(
                range(window - 1, len(history.history["val_accuracy"])),
                val_acc_smooth,
                label=f"Val Accuracy (smoothed {window})",
                linewidth=2,
                color="red",
            )
            axes[1, 2].plot(
                history.history["val_accuracy"],
                label="Val Accuracy (original)",
                linewidth=1,
                alpha=0.5,
                color="lightcoral",
            )
            axes[1, 2].set_title("Smoothed Validation Accuracy")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("Accuracy")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_with_confidence(self, image_path, threshold=0.7):
        """Make predictions with confidence analysis"""
        if self.model is None:
            return None

        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return None

        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image, verbose=0)

        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [(i, predictions[0][i]) for i in top_5_indices]

        result = {
            "predicted_class": (
                self.class_names[predicted_class]
                if predicted_class < len(self.class_names)
                else f"Class_{predicted_class}"
            ),
            "confidence": float(confidence),
            "class_id": int(predicted_class),
            "is_confident": confidence > threshold,
            "top_5_predictions": [
                {
                    "class": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                    "confidence": float(conf),
                    "class_id": int(i),
                }
                for i, conf in top_5_predictions
            ],
            "entropy": float(-np.sum(predictions[0] * np.log(predictions[0] + 1e-8))),
        }

        return result

    def save_model(self, filepath="Models/furniture_model_final.h5"):
        """Save the final trained model with metadata"""
        if self.model:
            os.makedirs("Models", exist_ok=True)
            self.model.save(filepath)

            # Save metadata
            metadata = {
                "class_names": self.class_names,
                "img_size": self.img_size,
                "num_classes": len(self.class_names),
                "model_type": "EfficientNetB0",
            }

            with open(filepath.replace(".h5", "_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Final model and metadata saved to: {filepath}")

    def load_model(self, filepath="Models/furniture_model_final.h5"):
        """Load a previously trained model"""
        self.model = tf.keras.models.load_model(filepath)

        # Load metadata
        metadata_path = filepath.replace(".h5", "_metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.class_names = metadata.get("class_names", [])
                self.img_size = tuple(metadata.get("img_size", (224, 224)))
            print(f"Model and metadata loaded from: {filepath}")
        except FileNotFoundError:
            print("Metadata file not found, using default values")


def main():
    """Main training pipeline"""
    # Initialize model
    furniture_model = ImprovedFurnitureModel(img_size=(224, 224), batch_size=16)

    # Download and prepare dataset
    gdrive_file_id = "1i3cNtxQ0xZTn2-ytDYMLpmYEgBGSI3UP"
    dataset_path = furniture_model.download_and_extract_remote_dataset(gdrive_file_id)
    train_data, val_data = furniture_model.parse_yolo_annotations(dataset_path)

    print(f"Dataset prepared: {len(train_data[0])} training, {len(val_data[0])} validation samples")

    # Create data generators (no augmentation since dataset already includes it)
    train_generator, val_generator = furniture_model.create_data_generators(train_data, val_data)

    # Build model
    model = furniture_model.build_model(num_classes=len(furniture_model.class_names))
    print(f"Model parameters: {model.count_params():,}")
    print(f"Number of classes: {len(furniture_model.class_names)}")

    # Train model with single approach
    history = furniture_model.train_model(train_generator, val_generator, epochs=80)

    # Analyze and visualize results
    furniture_model.plot_training_history(history)
    furniture_model.evaluate_model_performance(history)

    # Save final model
    furniture_model.save_model("Models/furniture_model_final.h5")

    if os.path.exists("dataset/"):
        shutil.rmtree("dataset/")
        print("Temporary dataset cleaned up")


if __name__ == "__main__":
    main()
