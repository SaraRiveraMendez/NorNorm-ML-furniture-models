import datetime
import json
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from ultralytics import YOLO


class PureYOLOv12FurnitureClassifier:
    def __init__(self, model_size="n", img_size=640, batch_size=16):
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.class_weights = None
        self.progressive_unfreeze_schedule = []

        # Create folder with timestamp for this training session
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/YOLOv12_Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def convert_segmentation_to_detection(self, label_dir):
        """Convert any segmentation annotations to detection by creating bounding boxes"""
        converted_count = 0

        for label_file in os.listdir(label_dir):
            if label_file.endswith(".txt"):
                file_path = os.path.join(label_dir, label_file)
                new_lines = []
                needs_conversion = False

                with open(file_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 5:  # This is segmentation
                        needs_conversion = True
                        # Convert segmentation points to bounding box
                        class_id = parts[0]
                        points = list(map(float, parts[1:]))

                        # Get all x and y coordinates
                        x_coords = points[0::2]  # x values
                        y_coords = points[1::2]  # y values

                        # Calculate bounding box coordinates
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # Convert to YOLO detection format
                        width = x_max - x_min
                        height = y_max - y_min
                        center_x = x_min + width / 2
                        center_y = y_min + height / 2

                        new_line = (
                            f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                        )
                        new_lines.append(new_line)
                        converted_count += 1
                    else:
                        # Already in detection format, keep as is
                        new_lines.append(line)

                if needs_conversion:
                    # Write the converted file
                    with open(file_path, "w") as f:
                        f.writelines(new_lines)
                    print(f"Converted {label_file}")

        print(f"Total annotations converted: {converted_count}")
        return converted_count

    def download_and_extract_dataset(self, gdrive_file_id, output_filename=None):
        """Download dataset from Google Drive and extract"""
        import tempfile

        if output_filename is None:
            output_filename = f"dataset_{gdrive_file_id}.zip"

        url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, output_filename)

            print("Downloading dataset from Google Drive...")
            gdown.download(url, temp_zip_path, quiet=False)

            extract_path = "dataset/"
            print("Extracting dataset...")
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

        return extract_path

    def prepare_classification_dataset(self, dataset_path, train_split=0.8):
        """Extract ALL elements from images and split into train/validation 80/20"""
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])
                print(f"Classes found: {self.class_names}")

        # Convert segmentation annotations to detection format
        print("Converting segmentation annotations to detection format...")
        for split in ["train", "val"]:
            labels_dir = os.path.join(dataset_path, split, "labels")
            if os.path.exists(labels_dir):
                converted = self.convert_segmentation_to_detection(labels_dir)
                print(f"Converted {converted} annotations in {split} set")

        all_images = []
        all_labels = []
        total_elements = 0
        images_processed = 0

        # Process all splits (train and valid) together
        for split in ["train", "val"]:
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
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])

                                        # Verify valid class_id
                                        if 0 <= class_id < len(self.class_names):
                                            all_images.append(img_path)
                                            all_labels.append(class_id)
                                            elements_in_image += 1
                                            total_elements += 1

                            if elements_in_image > 0:
                                images_processed += 1

        print(f"Images processed: {images_processed}")
        print(f"Total elements extracted: {total_elements}")
        print(f"Average elements per image: {total_elements/images_processed:.2f}")

        # Calculate class weights for balancing
        unique_labels = np.unique(all_labels)
        class_weights_array = compute_class_weight("balanced", classes=unique_labels, y=all_labels)
        self.class_weights = dict(zip(unique_labels, class_weights_array))

        print(f"\nClass weights calculated for balancing:")
        for class_id, weight in self.class_weights.items():
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"Unknown_{class_id}"
            )
            print(f"  {class_name}: {weight:.3f}")

        # Split data 80/20
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images,
            all_labels,
            test_size=(1 - train_split),
            random_state=42,
            stratify=all_labels,
        )

        print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")

        # Show class distribution
        print(f"\nClass distribution in training:")
        unique, counts = np.unique(train_labels, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"Unknown_{class_id}"
            )
            print(f"  {class_name}: {count} samples")

        print(f"\nClass distribution in validation:")
        unique, counts = np.unique(val_labels, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"Unknown_{class_id}"
            )
            print(f"  {class_name}: {count} samples")

        return (train_images, train_labels), (val_images, val_labels)

    # ... (rest of your class methods remain the same) ...


def main():
    """Main training pipeline for pure YOLOv12 classification"""
    # Initialize classifier
    classifier = PureYOLOv12FurnitureClassifier(model_size="n", img_size=640, batch_size=16)

    # Download and prepare dataset
    print("Step 1: Downloading dataset...")
    gdrive_file_id = "1Yyp12TpZY8OggZVmkU6JrokJm3xIzQto"
    dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

    # Prepare classification dataset (no cropping needed)
    print("\nStep 2: Preparing classification dataset...")
    (train_images, train_labels), (val_images, val_labels) = (
        classifier.prepare_classification_dataset(dataset_path, train_split=0.8)
    )

    # Initialize model
    print("\nStep 3: Initializing YOLOv12 model...")
    if not classifier.initialize_yolov12_classifier():
        print("Failed to initialize model")
        return

    # Train model with progressive unfreezing
    print("\nStep 4: Training model with progressive unfreezing...")
    training_results = classifier.train_model(dataset_path, epochs=100)
    print(training_results)

    # Validate model with Top-K metrics
    print("\nStep 5: Validating model with Top-K accuracy...")
    validation_results = classifier.validate_model(dataset_path)
    print(validation_results)

    # Create confusion matrix with Top-K metrics
    print("\nStep 6: Creating confusion matrix with Top-K metrics...")
    classifier.create_confusion_matrix(dataset_path)

    # Save model info
    print("\nStep 7: Saving model information...")
    classifier.save_model_info()

    # Cleanup temporary files
    if os.path.exists("dataset/") and dataset_path != "dataset/":
        shutil.rmtree("dataset/")

    print(f"\n{'='*60}")
    print("PURE YOLOV12 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Model saved in: {classifier.save_dir}")
    print("Features implemented:")
    print("✓ Progressive unfreezing of layers")
    print("✓ Top-1 and Top-3 accuracy metrics")
    print("✓ No image cropping (direct classification)")
    print("✓ Class balancing with weights")
    print("✓ Segmentation to detection conversion")
    print("\nFiles created:")
    print("- Best model: training/weights/best.pt")
    print("- Last model: training/weights/last.pt")
    print("- Training plots: training/")
    print("- Confusion matrix: confusion_matrix_yolov12.png")
    print("- Classification report: classification_report_yolov12.json")
    print("- Model metadata: model_metadata.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
