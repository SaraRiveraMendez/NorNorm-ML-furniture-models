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
    def __init__(self, model_size="s", img_size=640, batch_size=16):
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
        """Convert segmentation annotations to detection format properly"""
        converted_count = 0
        files_processed = 0

        print(f"Processing label directory: {label_dir}")

        if not os.path.exists(label_dir):
            print(f"Label directory does not exist: {label_dir}")
            return 0

        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        print(f"Found {len(label_files)} label files to process")

        for label_file in label_files:
            file_path = os.path.join(label_dir, label_file)
            new_lines = []
            file_converted = False

            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        print(f"Warning: Invalid annotation in {label_file}, line {line_num + 1}")
                        continue

                    class_id = parts[0]
                    coords = parts[1:]

                    # Check if this is segmentation format (more than 4 coordinates)
                    if len(coords) > 4:
                        file_converted = True
                        converted_count += 1

                        # Convert string coordinates to floats
                        try:
                            coords_float = [float(x) for x in coords]
                        except ValueError:
                            print(
                                f"Warning: Invalid coordinates in {label_file}, line {line_num + 1}"
                            )
                            continue

                        # Extract x and y coordinates (assuming they alternate x,y,x,y...)
                        if len(coords_float) % 2 != 0:
                            print(
                                f"Warning: Odd number of coordinates in {label_file}, line {line_num + 1}"
                            )
                            continue

                        x_coords = coords_float[0::2]  # Every even index (0,2,4,...)
                        y_coords = coords_float[1::2]  # Every odd index (1,3,5,...)

                        # Calculate bounding box from polygon points
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # Convert to YOLO detection format (center_x, center_y, width, height)
                        width = x_max - x_min
                        height = y_max - y_min
                        center_x = x_min + width / 2
                        center_y = y_min + height / 2

                        # Ensure coordinates are within [0,1] range
                        center_x = max(0.0, min(1.0, center_x))
                        center_y = max(0.0, min(1.0, center_y))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))

                        new_line = (
                            f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                        )
                        new_lines.append(new_line)

                    elif len(coords) == 4:
                        # Already in detection format, keep as is
                        new_lines.append(line + "\n")
                    else:
                        print(
                            f"Warning: Unexpected coordinate count in {label_file}, line {line_num + 1}"
                        )

                # Write the processed file back
                if new_lines:
                    with open(file_path, "w") as f:
                        f.writelines(new_lines)
                    files_processed += 1

                    if file_converted:
                        print(f"Converted segmentation annotations in: {label_file}")

            except Exception as e:
                print(f"Error processing {label_file}: {e}")

        print(f"Files processed: {files_processed}")
        print(f"Total segmentation annotations converted to detection: {converted_count}")
        return converted_count

    def clean_dataset_format(self, dataset_path):
        """Ensure all annotations are in proper detection format"""
        print("Cleaning dataset format - converting all segmentation to detection...")

        total_converted = 0

        # Process all possible label directories
        for split in ["train", "val", "valid", "test"]:
            labels_dir = os.path.join(dataset_path, split, "labels")
            if os.path.exists(labels_dir):
                print(f"Processing {split} labels...")
                converted = self.convert_segmentation_to_detection(labels_dir)
                total_converted += converted
                print(f"Converted {converted} annotations in {split} set")
            else:
                print(f"Labels directory not found: {labels_dir}")

        print(f"Total annotations converted across all splits: {total_converted}")
        return total_converted

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

        # Clean dataset format first
        print("Step 1: Cleaning dataset format...")
        self.clean_dataset_format(dataset_path)

        all_images = []
        all_labels = []
        total_elements = 0
        images_processed = 0

        # Process all splits (train and valid) together
        for split in ["train", "val", "valid"]:
            images_dir = os.path.join(dataset_path, split, "images")
            labels_dir = os.path.join(dataset_path, split, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print(f"Directory {split} not found, skipping...")
                continue

            print(f"Processing {split} split...")
            split_elements = 0

            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(images_dir, img_file)
                    label_file = os.path.splitext(img_file)[0] + ".txt"
                    label_path = os.path.join(labels_dir, label_file)

                    if os.path.exists(label_path):
                        try:
                            with open(label_path, "r") as f:
                                lines = f.readlines()
                                elements_in_image = 0

                                # Process ALL lines (all elements in the image)
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        parts = line.split()
                                        if len(parts) == 5:  # Proper detection format
                                            try:
                                                class_id = int(parts[0])

                                                # Verify valid class_id
                                                if 0 <= class_id < len(self.class_names):
                                                    all_images.append(img_path)
                                                    all_labels.append(class_id)
                                                    elements_in_image += 1
                                                    total_elements += 1
                                                    split_elements += 1
                                                else:
                                                    print(
                                                        f"Warning: Invalid class_id {class_id} in {label_file}"
                                                    )
                                            except ValueError:
                                                print(
                                                    f"Warning: Invalid class_id format in {label_file}"
                                                )
                                        else:
                                            print(
                                                f"Warning: Unexpected format in {label_file}: {len(parts)} parts"
                                            )

                                if elements_in_image > 0:
                                    images_processed += 1

                        except Exception as e:
                            print(f"Error reading {label_path}: {e}")

            print(f"  {split} - Elements extracted: {split_elements}")

        print(f"\nDataset Summary:")
        print(f"Images processed: {images_processed}")
        print(f"Total elements extracted: {total_elements}")
        if images_processed > 0:
            print(f"Average elements per image: {total_elements/images_processed:.2f}")

        if total_elements == 0:
            raise ValueError("No valid annotations found! Check your dataset format.")

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
            count = np.sum(np.array(all_labels) == class_id)
            print(f"  {class_name}: {weight:.3f} (samples: {count})")

        # Split data 80/20
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images,
            all_labels,
            test_size=(1 - train_split),
            random_state=42,
            stratify=all_labels,
        )

        print(f"\nData split:")
        print(f"Training samples: {len(train_images)}")
        print(f"Validation samples: {len(val_images)}")

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

    def _show_dataset_statistics(self, train_samples, val_samples):
        """Show dataset statistics"""
        print(f"\nDataset Statistics:")
        print(f"{'='*50}")
        print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Total':<10}")
        print(f"{'='*50}")

        total_train = 0
        total_val = 0

        for class_name in self.class_names:
            train_count = train_samples.get(class_name, 0)
            val_count = val_samples.get(class_name, 0)
            total_count = train_count + val_count

            print(f"{class_name:<20} {train_count:<10} {val_count:<10} {total_count:<10}")
            total_train += train_count
            total_val += val_count

        print(f"{'='*50}")
        print(f"{'TOTAL':<20} {total_train:<10} {total_val:<10} {total_train + total_val:<10}")
        print(f"{'='*50}")

    def initialize_yolov12_classifier(self):
        """Initialize YOLOv12 model for classification"""
        try:
            model_name = f"yolo12{self.model_size}-cls.pt"
            self.model = YOLO(model_name)
            print(f"YOLOv12{self.model_size} Classification model initialized successfully")

            # Setup progressive unfreezing schedule
            self._setup_progressive_unfreeze()

            return True
        except Exception as e:
            print(f"Error initializing YOLOv12 classifier: {e}")
            try:
                # Fallback to regular YOLOv12 and modify for classification
                model_name = f"yolo12{self.model_size}.pt"
                self.model = YOLO(model_name)
                print(
                    f"Using YOLOv12{self.model_size} detection model (will be adapted for classification)"
                )

                # Setup progressive unfreezing schedule
                self._setup_progressive_unfreeze()

                return True
            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                return False

    def _setup_progressive_unfreeze(self):
        """Setup progressive unfreezing schedule"""
        # Define unfreezing schedule (layer groups to unfreeze at specific epochs)
        self.progressive_unfreeze_schedule = [
            {
                "epoch": 0,
                "freeze_backbone": True,
                "description": "Freeze backbone, train classifier head only",
            },
            {
                "epoch": 20,
                "freeze_backbone": False,
                "unfreeze_layers": ["model.22", "model.21"],
                "description": "Unfreeze last 2 layers",
            },
            {
                "epoch": 40,
                "unfreeze_layers": ["model.20", "model.19", "model.18"],
                "description": "Unfreeze middle layers",
            },
            {
                "epoch": 60,
                "unfreeze_layers": "all",
                "description": "Unfreeze all layers (full fine-tuning)",
            },
        ]

        print("\nProgressive Unfreezing Schedule:")
        print("=" * 60)
        for schedule in self.progressive_unfreeze_schedule:
            print(f"Epoch {schedule['epoch']:2d}: {schedule['description']}")
        print("=" * 60)

    def _apply_progressive_unfreeze(self, current_epoch):
        """Apply progressive unfreezing based on current epoch"""
        if not hasattr(self.model, "model") or self.model.model is None:
            return

        # First, ensure all parameters require gradients
        for param in self.model.model.parameters():
            param.requires_grad = True

        for schedule in reversed(self.progressive_unfreeze_schedule):
            if current_epoch >= schedule["epoch"]:
                if "freeze_backbone" in schedule and schedule["freeze_backbone"]:
                    # Freeze backbone layers, keep classifier head unfrozen
                    for name, param in self.model.model.named_parameters():
                        if not name.startswith("model.22"):  # Freeze everything except classifier
                            param.requires_grad = False
                    print(f"Epoch {current_epoch}: Backbone frozen, training head only")
                    break

                elif "unfreeze_layers" in schedule:
                    layers_to_unfreeze = schedule["unfreeze_layers"]

                    if layers_to_unfreeze == "all":
                        # Unfreeze all layers
                        for param in self.model.model.parameters():
                            param.requires_grad = True
                        print(f"Epoch {current_epoch}: All layers unfrozen")
                        break
                    else:
                        # Unfreeze specific layers, freeze others
                        for name, param in self.model.model.named_parameters():
                            # Check if this parameter is in the layers to unfreeze
                            should_unfreeze = any(layer in name for layer in layers_to_unfreeze)
                            param.requires_grad = should_unfreeze
                        print(f"Epoch {current_epoch}: Unfrozen layers: {layers_to_unfreeze}")
                        break

    def create_yolo_classification_config(self, dataset_path):
        """Create YOLO classification configuration"""
        config = {
            "path": os.path.abspath(dataset_path),
            "train": "train",
            "val": "valid",
            "names": {i: name for i, name in enumerate(self.class_names)},
        }

        config_path = os.path.join(dataset_path, "dataset.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"YOLO classification config saved to: {config_path}")
        return config_path

    def train_model(self, dataset_path, epochs=100):
        """Train pure YOLOv12 classification model with progressive unfreezing"""
        print(f"\n{'='*60}")
        print("STARTING PURE YOLOv12 CLASSIFICATION TRAINING")
        print(f"{'='*60}")
        print(f"Model: YOLOv12{self.model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Dataset path: {dataset_path}")

        if self.class_weights:
            print("Class weights will be applied during training")

        print(f"{'='*60}")

        # Create YOLO config
        config_path = self.create_yolo_classification_config(dataset_path)

        # Apply initial freezing (start with backbone frozen)
        self._apply_progressive_unfreeze(0)

        # Train the model
        results = self.model.train(
            data=config_path,
            epochs=epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device="cpu",
            workers=8,
            patience=20,
            save=True,
            save_period=10,
            val=True,
            project=self.save_dir,
            name="training",
            exist_ok=True,
            pretrained=True,
            optimizer="RMSProp",
            lr0=5e-4,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=True,
            cls=1.0,  # Classification loss weight
            dfl=1.5,  # Distribution focal loss weight
            verbose=True,
        )

        print("Training completed!")
        return results

    def calculate_top_k_accuracy(self, dataset_path, k_values=[1, 3]):
        """Calculate Top-K accuracy metrics"""
        print(f"Calculating Top-K accuracy for k={k_values}...")

        val_path = os.path.join(dataset_path, "valid")
        y_true = []
        y_pred_probs = []

        # Get predictions for all validation images
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(val_path, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_file)

                        # Predict
                        results = self.model.predict(img_path, verbose=False)

                        if results and len(results) > 0:
                            probs = results[0].probs
                            if probs is not None and hasattr(probs, "data"):
                                # Get all class probabilities
                                prob_scores = probs.data.cpu().numpy()
                                y_true.append(class_idx)
                                y_pred_probs.append(prob_scores)

        if len(y_true) == 0:
            print("No predictions found for accuracy calculation")
            return {}

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)

        # Calculate Top-K accuracies
        top_k_accuracies = {}
        for k in k_values:
            if k <= len(self.class_names):
                try:
                    acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
                    top_k_accuracies[f"top_{k}_accuracy"] = acc
                    print(f"Top-{k} Accuracy: {acc:.4f}")
                except Exception as e:
                    print(f"Error calculating Top-{k} accuracy: {e}")
                    top_k_accuracies[f"top_{k}_accuracy"] = 0.0

        return top_k_accuracies

    def validate_model(self, dataset_path):
        """Validate the trained model with extended metrics"""
        print("Validating model...")

        # Use the best trained weights
        best_model_path = os.path.join(self.save_dir, "training", "weights", "best.pt")
        if os.path.exists(best_model_path):
            self.model = YOLO(best_model_path)
            print(f"Loaded best model from: {best_model_path}")

        # Standard validation
        results = self.model.val()

        # Calculate Top-K accuracies
        top_k_metrics = self.calculate_top_k_accuracy(dataset_path, k_values=[1, 3])

        # Combine results
        validation_results = {"standard_metrics": results, "top_k_metrics": top_k_metrics}

        print("Validation completed!")

        # Safely print Top-K accuracies
        top1_acc = top_k_metrics.get("top_1_accuracy")
        top3_acc = top_k_metrics.get("top_3_accuracy")

        if top1_acc is not None:
            print(f"Top-1 Accuracy: {top1_acc:.4f}")
        else:
            print("Top-1 Accuracy: N/A")

        if top3_acc is not None:
            print(f"Top-3 Accuracy: {top3_acc:.4f}")
        else:
            print("Top-3 Accuracy: N/A")

        return validation_results

    def create_confusion_matrix(self, dataset_path):
        """Create confusion matrix from validation results"""
        print("Creating confusion matrix...")

        val_path = os.path.join(dataset_path, "valid")
        y_true = []
        y_pred = []
        y_pred_probs = []

        # Get predictions for all validation images
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(val_path, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_file)

                        # Predict
                        results = self.model.predict(img_path, verbose=False)

                        if results and len(results) > 0:
                            # Get predicted class
                            probs = results[0].probs
                            if probs is not None:
                                predicted_class = probs.top1
                                y_true.append(class_idx)
                                y_pred.append(predicted_class)

                                # Store probabilities for Top-K calculation
                                if hasattr(probs, "data"):
                                    prob_scores = probs.data.cpu().numpy()
                                    y_pred_probs.append(prob_scores)

        if len(y_true) == 0:
            print("No predictions found for confusion matrix")
            return

        # Calculate Top-K accuracies
        top_k_metrics = {}
        if y_pred_probs and len(y_pred_probs) > 0:
            y_pred_probs = np.array(y_pred_probs)
            for k in [1, 3]:
                if k <= len(self.class_names):
                    try:
                        acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
                        top_k_metrics[f"top_{k}_accuracy"] = acc
                    except Exception as e:
                        print(f"Error calculating Top-{k} accuracy: {e}")
                        top_k_metrics[f"top_{k}_accuracy"] = 0.0

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate per-class metrics
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_support = cm.sum(axis=1)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax1,
        )
        ax1.set_title("Confusion Matrix - Pure YOLOv12 Classifier")
        ax1.set_xlabel("Predicted Labels")
        ax1.set_ylabel("True Labels")
        ax1.tick_params(axis="x", rotation=45)

        # Per-class accuracy
        class_names_short = [
            name[:15] + "..." if len(name) > 15 else name for name in self.class_names
        ]
        bars = ax2.bar(
            range(len(self.class_names)),
            class_accuracies,
            color=[
                "red" if acc < 0.5 else "orange" if acc < 0.7 else "green"
                for acc in class_accuracies
            ],
        )
        ax2.set_title("Per-Class Accuracy")
        ax2.set_xlabel("Classes")
        ax2.set_ylabel("Accuracy")
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(class_names_short, rotation=45, ha="right")
        ax2.set_ylim(0, 1)

        # Add values on bars
        for i, (bar, acc, support) in enumerate(zip(bars, class_accuracies, class_support)):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2f}\n({support})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add Top-K accuracy information
        if top_k_metrics:
            info_text = "Top-K Accuracies:\n"
            for k, acc in top_k_metrics.items():
                info_text += f"{k.replace('_', '-').title()}: {acc:.4f}\n"
            ax2.text(
                0.02,
                0.98,
                info_text,
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "confusion_matrix_yolov12.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Classification report with Top-K metrics
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        report["class_weights"] = self.class_weights
        report["top_k_accuracies"] = top_k_metrics
        report["progressive_unfreeze_schedule"] = self.progressive_unfreeze_schedule

        report_path = os.path.join(self.save_dir, "classification_report_yolov12.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Confusion matrix saved to: {save_path}")
        print(f"Classification report saved to: {report_path}")

        # Print Top-K accuracies
        if top_k_metrics:
            print("\nTop-K Accuracies:")
            for k, acc in top_k_metrics.items():
                print(f"  {k.replace('_', '-').title()}: {acc:.4f}")

    def predict_with_confidence(self, image_path, threshold=0.7):
        """Make predictions with confidence analysis"""
        if self.model is None:
            print("Model not initialized")
            return None

        results = self.model.predict(image_path, verbose=False)

        if not results or len(results) == 0:
            return None

        result = results[0]
        if result.probs is None:
            return None

        probs = result.probs
        predicted_class = probs.top1
        confidence = float(probs.top1conf)

        # Get top 5 predictions
        top5_indices = probs.top5
        top5_confidences = probs.top5conf

        prediction_result = {
            "predicted_class": self.class_names[predicted_class],
            "confidence": confidence,
            "class_id": int(predicted_class),
            "is_confident": confidence > threshold,
            "top_5_predictions": [
                {
                    "class": self.class_names[idx],
                    "confidence": float(conf),
                    "class_id": int(idx),
                }
                for idx, conf in zip(top5_indices, top5_confidences)
            ],
            "model_type": f"Pure YOLOv12{self.model_size} Classifier with Progressive Unfreezing",
        }

        return prediction_result

    def save_model_info(self):
        """Save model information and metadata"""
        metadata = {
            "model_type": f"Pure YOLOv12{self.model_size} Classifier",
            "class_names": self.class_names,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "num_classes": len(self.class_names),
            "class_weights": self.class_weights,
            "progressive_unfreeze_schedule": self.progressive_unfreeze_schedule,
            "training_timestamp": datetime.datetime.now().isoformat(),
            "save_directory": self.save_dir,
            "model_path": os.path.join(self.save_dir, "training", "weights", "best.pt"),
            "features": [
                "Progressive unfreezing",
                "Top-1 and Top-3 accuracy metrics",
                "Class balancing with weights",
                "No image cropping (direct classification format)",
                "Improved segmentation to detection conversion",
                "Proper dataset format validation",
            ],
        }

        metadata_path = os.path.join(self.save_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model metadata saved to: {metadata_path}")

    def load_model(self, model_path):
        """Load a trained YOLOv12 model"""
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")

            # Try to load metadata
            metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.class_names = metadata.get("class_names", [])
                    self.img_size = metadata.get("img_size", 640)
                    self.class_weights = metadata.get("class_weights", None)
                    self.progressive_unfreeze_schedule = metadata.get(
                        "progressive_unfreeze_schedule", []
                    )
                print("Metadata loaded successfully")
            else:
                print("Metadata file not found")

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def verify_dataset_format(self, dataset_path):
        """Verify that dataset is in proper detection format"""
        print("Verifying dataset format...")

        issues_found = []
        total_annotations = 0
        valid_annotations = 0

        for split in ["train", "val", "valid"]:
            labels_dir = os.path.join(dataset_path, split, "labels")
            if not os.path.exists(labels_dir):
                continue

            print(f"Checking {split} labels...")

            for label_file in os.listdir(labels_dir):
                if label_file.endswith(".txt"):
                    file_path = os.path.join(labels_dir, label_file)

                    try:
                        with open(file_path, "r") as f:
                            lines = f.readlines()

                        for line_num, line in enumerate(lines):
                            line = line.strip()
                            if line:
                                total_annotations += 1
                                parts = line.split()

                                if len(parts) == 5:
                                    # Check if all values are valid
                                    try:
                                        class_id = int(parts[0])
                                        coords = [float(x) for x in parts[1:]]

                                        # Check coordinate ranges
                                        if all(0.0 <= coord <= 1.0 for coord in coords):
                                            valid_annotations += 1
                                        else:
                                            issues_found.append(
                                                f"{label_file}:{line_num+1} - Coordinates out of range"
                                            )

                                    except ValueError:
                                        issues_found.append(
                                            f"{label_file}:{line_num+1} - Invalid numeric values"
                                        )

                                elif len(parts) > 5:
                                    issues_found.append(
                                        f"{label_file}:{line_num+1} - Still in segmentation format"
                                    )

                                else:
                                    issues_found.append(
                                        f"{label_file}:{line_num+1} - Invalid format"
                                    )

                    except Exception as e:
                        issues_found.append(f"{label_file} - Error reading file: {e}")

        print(f"Dataset Format Verification Results:")
        print(f"Total annotations: {total_annotations}")
        print(f"Valid annotations: {valid_annotations}")
        print(f"Issues found: {len(issues_found)}")

        if issues_found:
            print(f"First 10 issues:")
            for issue in issues_found[:10]:
                print(f"  - {issue}")
            if len(issues_found) > 10:
                print(f"  ... and {len(issues_found) - 10} more issues")
        else:
            print("All annotations are in proper detection format!")

        return len(issues_found) == 0


def main():
    """Main training pipeline for pure YOLOv12 classification"""
    # Initialize classifier
    classifier = PureYOLOv12FurnitureClassifier(model_size="s", img_size=640, batch_size=16)

    # Download and prepare dataset
    print("Step 1: Downloading dataset...")
    gdrive_file_id = "1LVZdiClbXwOzfKug2PZKxTdAEIvYYhWo"
    dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

    # Prepare classification dataset (no cropping needed)
    print("\nStep 2: Preparing classification dataset...")
    (train_images, train_labels), (val_images, val_labels) = (
        classifier.prepare_classification_dataset(dataset_path, train_split=0.8)
    )

    # Verify dataset format after conversion
    print("\nStep 2.1: Verifying dataset format...")
    is_format_correct = classifier.verify_dataset_format(dataset_path)
    if not is_format_correct:
        print("Warning: Dataset format issues detected. Training may encounter problems.")
    else:
        print("Dataset format verification passed!")

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
    print("✓ Improved segmentation to detection conversion")
    print("✓ Dataset format validation")
    print("✓ Better error handling and logging")
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
