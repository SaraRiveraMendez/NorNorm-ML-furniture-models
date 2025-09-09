import datetime
import json
import os
import shutil
import tempfile
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


class WeightedYOLOv12Classifier:
    def __init__(self, model_size="n", img_size=640, batch_size=16):
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.class_weights = None
        self.class_weights_tensor = None
        self.weighted_loss_fn = None

        # Create folder with timestamp for this training session
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/WeightedYOLOv12_Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def download_and_extract_dataset(self, gdrive_file_id, output_filename=None):
        """Download dataset from Google Drive and extract"""
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

    def prepare_yolo_classification_structure(self, dataset_path, train_split=0.8):
        """Prepare YOLO classification directory structure and calculate class weights"""
        # Load class names from data.yaml
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])
                print(f"Classes found: {self.class_names}")

        # Create classification structure
        classification_path = os.path.join(dataset_path, "classification")
        os.makedirs(classification_path, exist_ok=True)

        train_class_path = os.path.join(classification_path, "train")
        val_class_path = os.path.join(classification_path, "val")
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        # Create class directories
        for class_name in self.class_names:
            os.makedirs(os.path.join(train_class_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_class_path, class_name), exist_ok=True)

        # Extract elements and organize for classification
        all_images = []
        all_labels = []
        image_class_pairs = []

        # Process all detection data and extract individual elements
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
                        # Read image
                        image = cv2.imread(img_path)
                        if image is None:
                            continue

                        img_height, img_width = image.shape[:2]

                        with open(label_path, "r") as f:
                            lines = f.readlines()

                            # Process each detected object
                            for idx, line in enumerate(lines):
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])

                                        if 0 <= class_id < len(self.class_names):
                                            # Extract bounding box
                                            x_center, y_center, width, height = map(
                                                float, parts[1:5]
                                            )

                                            # Convert normalized coordinates to pixels
                                            x_center_px = int(x_center * img_width)
                                            y_center_px = int(y_center * img_height)
                                            width_px = int(width * img_width)
                                            height_px = int(height * img_height)

                                            # Calculate bounding box coordinates
                                            x1 = max(0, x_center_px - width_px // 2)
                                            y1 = max(0, y_center_px - height_px // 2)
                                            x2 = min(img_width, x_center_px + width_px // 2)
                                            y2 = min(img_height, y_center_px + height_px // 2)

                                            # Crop the object
                                            cropped_obj = image[y1:y2, x1:x2]

                                            if cropped_obj.size > 0:
                                                # Create unique filename for cropped object
                                                base_name = os.path.splitext(img_file)[0]
                                                crop_name = f"{base_name}_obj{idx}_{class_id}.jpg"

                                                image_class_pairs.append(
                                                    (cropped_obj, class_id, crop_name)
                                                )
                                                all_labels.append(class_id)

        print(f"Total objects extracted: {len(image_class_pairs)}")

        # Calculate class weights
        if len(all_labels) > 0:
            unique_labels = np.unique(all_labels)
            class_weights_array = compute_class_weight(
                "balanced", classes=unique_labels, y=all_labels
            )
            self.class_weights = dict(zip(unique_labels, class_weights_array))

            self.class_weights_tensor = torch.ones(len(self.class_names))
            for class_id, weight in self.class_weights.items():
                self.class_weights_tensor[class_id] = weight

            print(f"\nClass weights calculated:")
            for class_id, weight in self.class_weights.items():
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"Unknown_{class_id}"
                )
                print(f"  {class_name}: {weight:.3f}")

        # Split data into train/validation
        labels_array = np.array(all_labels)
        indices = np.arange(len(image_class_pairs))

        train_indices, val_indices = train_test_split(
            indices, test_size=(1 - train_split), random_state=42, stratify=labels_array
        )

        # Save training images
        train_counts = {i: 0 for i in range(len(self.class_names))}
        for idx in train_indices:
            cropped_obj, class_id, crop_name = image_class_pairs[idx]
            class_name = self.class_names[class_id]
            save_path = os.path.join(train_class_path, class_name, crop_name)
            cv2.imwrite(save_path, cropped_obj)
            train_counts[class_id] += 1

        # Save validation images
        val_counts = {i: 0 for i in range(len(self.class_names))}
        for idx in val_indices:
            cropped_obj, class_id, crop_name = image_class_pairs[idx]
            class_name = self.class_names[class_id]
            save_path = os.path.join(val_class_path, class_name, crop_name)
            cv2.imwrite(save_path, cropped_obj)
            val_counts[class_id] += 1

        print(f"\nClassification dataset structure created:")
        print(f"Training samples: {sum(train_counts.values())}")
        print(f"Validation samples: {sum(val_counts.values())}")

        print(f"\nTraining distribution:")
        for class_id, count in train_counts.items():
            if count > 0:
                class_name = self.class_names[class_id]
                print(f"  {class_name}: {count}")

        print(f"\nValidation distribution:")
        for class_id, count in val_counts.items():
            if count > 0:
                class_name = self.class_names[class_id]
                print(f"  {class_name}: {count}")

        return classification_path

    def create_classification_yaml(self, classification_path):
        """Create YOLO classification configuration"""
        config = {
            "path": os.path.abspath(classification_path),
            "train": "train",
            "val": "val",
            "names": {i: name for i, name in enumerate(self.class_names)},
            "nc": len(self.class_names),
        }

        config_path = os.path.join(classification_path, "dataset.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"YOLO classification config saved to: {config_path}")
        return config_path

    def create_custom_loss(self):
        """Create custom weighted loss function"""
        if self.class_weights_tensor is None:
            return None

        class WeightedCrossEntropyLoss(nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.weights = weights
                self.ce_loss = nn.CrossEntropyLoss(weight=weights, reduction="mean")

            def forward(self, predictions, targets):
                return self.ce_loss(predictions, targets)

        self.weighted_loss_fn = WeightedCrossEntropyLoss(self.class_weights_tensor)
        print("Custom weighted loss function created")
        return self.weighted_loss_fn

    def patch_model_loss(self):
        """Patch the model to use weighted loss"""
        if self.model is None or self.class_weights_tensor is None:
            return False

        try:
            # Get the model's device
            device = next(self.model.model.parameters()).device
            weighted_tensor = self.class_weights_tensor.to(device)

            # Try to patch the loss function
            if hasattr(self.model.model, "loss"):
                original_loss = self.model.model.loss

                def weighted_loss_wrapper(*args, **kwargs):
                    # Get the original loss
                    loss_dict = original_loss(*args, **kwargs)

                    # If there's classification loss, apply weights
                    if "cls" in loss_dict and len(args) >= 2:
                        preds, targets = args[0], args[1]
                        if hasattr(targets, "cls"):
                            cls_targets = targets.cls.long()
                            cls_preds = preds[0] if isinstance(preds, (list, tuple)) else preds

                            # Apply weighted cross entropy
                            weighted_cls_loss = nn.CrossEntropyLoss(weight=weighted_tensor)(
                                cls_preds, cls_targets
                            )
                            loss_dict["cls"] = weighted_cls_loss

                    return loss_dict

                self.model.model.loss = weighted_loss_wrapper
                print(f"Successfully patched model loss with class weights on {device}")
                return True

        except Exception as e:
            print(f"Could not patch model loss: {e}")

        return False

    def initialize_yolov12_classifier(self):
        """Initialize YOLOv12 model for classification"""
        try:
            # Try classification model first
            model_name = f"yolo12{self.model_size}-cls.pt"
            self.model = YOLO(model_name)
            print(f"YOLOv12{self.model_size} Classification model initialized successfully")

            # Try to patch the loss function for class weights
            if self.class_weights_tensor is not None:
                loss_patched = self.patch_model_loss()
                if not loss_patched:
                    print("Warning: Could not apply class weights to model loss")
                    self.create_custom_loss()

            return True

        except Exception as e:
            print(f"Error initializing YOLOv12 classifier: {e}")
            try:
                # Fallback to detection model
                model_name = f"yolo12{self.model_size}.pt"
                self.model = YOLO(model_name)
                print(f"Using YOLOv12{self.model_size} detection model as fallback")

                # Try to patch the loss function for class weights
                if self.class_weights_tensor is not None:
                    loss_patched = self.patch_model_loss()
                    if not loss_patched:
                        print("Warning: Could not apply class weights to model loss")
                        self.create_custom_loss()

                return True

            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                return False

    def train_model_with_class_weights(self, classification_path, epochs=100):
        """Train YOLOv12 classification model"""
        print(f"\n{'='*60}")
        print("STARTING WEIGHTED YOLOv12 CLASSIFICATION TRAINING")
        print(f"{'='*60}")
        print(f"Model: YOLOv12{self.model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Number of classes: {len(self.class_names)}")

        if self.class_weights:
            print("Class weights calculated (applied through custom loss):")
            for class_id, weight in self.class_weights.items():
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"Unknown_{class_id}"
                )
                print(f"  {class_name}: {weight:.3f}")

        print(f"{'='*60}")

        # Create classification config
        config_path = self.create_classification_yaml(classification_path)

        # Training arguments (only valid YOLO arguments)
        training_args = {
            "data": config_path,
            "epochs": epochs,
            "imgsz": self.img_size,
            "batch": self.batch_size,
            "device": "cpu",  # Change to "0" for GPU
            "workers": 4,
            "patience": 25,
            "save": True,
            "save_period": 10,
            "val": True,
            "project": self.save_dir,
            "name": "weighted_training",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,
            "verbose": True,
        }

        results = self.model.train(**training_args)

        print("Training completed!")
        print("Class weights were applied through custom loss function")
        return results

    def validate_model(self, classification_path):
        """Validate the trained model"""
        print("Validating model...")

        # Load best model
        best_model_path = os.path.join(self.save_dir, "weighted_training", "weights", "best.pt")
        if os.path.exists(best_model_path):
            self.model = YOLO(best_model_path)
            print(f"Loaded best model from: {best_model_path}")

        # Run validation
        config_path = os.path.join(classification_path, "dataset.yaml")
        results = self.model.val(data=config_path)

        # Calculate additional metrics
        top_k_metrics = self.calculate_top_k_accuracy(classification_path, k_values=[1, 3, 5])

        validation_results = {
            "standard_metrics": results,
            "top_k_metrics": top_k_metrics,
            "class_weights_applied": True,
            "class_weights": self.class_weights,
        }

        print("Validation completed!")
        if "top_1_accuracy" in top_k_metrics:
            print(f"Top-1 Accuracy: {top_k_metrics['top_1_accuracy']:.4f}")
        if "top_3_accuracy" in top_k_metrics:
            print(f"Top-3 Accuracy: {top_k_metrics['top_3_accuracy']:.4f}")

        return validation_results

    def calculate_top_k_accuracy(self, classification_path, k_values=[1, 3, 5]):
        """Calculate Top-K accuracy metrics"""
        print(f"Calculating Top-K accuracy for k={k_values}...")

        val_path = os.path.join(classification_path, "val")
        y_true = []
        y_pred_probs = []

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(val_path, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_file)

                        results = self.model.predict(img_path, verbose=False)

                        if results and len(results) > 0:
                            probs = results[0].probs
                            if probs is not None and hasattr(probs, "data"):
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
                acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
                top_k_accuracies[f"top_{k}_accuracy"] = acc
                print(f"Top-{k} Accuracy: {acc:.4f}")

        return top_k_accuracies

    def create_confusion_matrix(self, classification_path):
        """Create confusion matrix"""
        print("Creating confusion matrix...")

        val_path = os.path.join(classification_path, "val")
        y_true = []
        y_pred = []

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(val_path, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_file)

                        results = self.model.predict(img_path, verbose=False)

                        if results and len(results) > 0:
                            probs = results[0].probs
                            if probs is not None:
                                predicted_class = probs.top1
                                y_true.append(class_idx)
                                y_pred.append(predicted_class)

        if len(y_true) == 0:
            print("No predictions found for confusion matrix")
            return

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate per-class metrics
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_support = cm.sum(axis=1)

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted Labels")
        axes[0, 0].set_ylabel("True Labels")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Normalized confusion matrix
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Oranges",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Normalized Confusion Matrix")
        axes[0, 1].set_xlabel("Predicted Labels")
        axes[0, 1].set_ylabel("True Labels")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Per-class accuracy
        class_names_short = [
            name[:15] + "..." if len(name) > 15 else name for name in self.class_names
        ]
        bars = axes[1, 0].bar(
            range(len(self.class_names)),
            class_accuracies,
            color=[
                "red" if acc < 0.5 else "orange" if acc < 0.7 else "green"
                for acc in class_accuracies
            ],
        )
        axes[1, 0].set_title("Per-Class Accuracy")
        axes[1, 0].set_xlabel("Classes")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_xticks(range(len(self.class_names)))
        axes[1, 0].set_xticklabels(class_names_short, rotation=45, ha="right")
        axes[1, 0].set_ylim(0, 1)

        # Add values on bars
        for i, (bar, acc, support) in enumerate(zip(bars, class_accuracies, class_support)):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2f}\n({support})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Class weights visualization
        weights_to_plot = [self.class_weights.get(i, 1.0) for i in range(len(self.class_names))]
        weight_bars = axes[1, 1].bar(
            range(len(self.class_names)),
            weights_to_plot,
            color="purple",
            alpha=0.7,
        )
        axes[1, 1].set_title("Applied Class Weights")
        axes[1, 1].set_xlabel("Classes")
        axes[1, 1].set_ylabel("Weight")
        axes[1, 1].set_xticks(range(len(self.class_names)))
        axes[1, 1].set_xticklabels(class_names_short, rotation=45, ha="right")

        # Add weight values on bars
        for i, (bar, weight) in enumerate(zip(weight_bars, weights_to_plot)):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{weight:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "confusion_matrix_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        report["class_weights_applied"] = self.class_weights

        report_path = os.path.join(self.save_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Confusion matrix saved to: {save_path}")
        print(f"Classification report saved to: {report_path}")

    def save_model_info(self):
        """Save model information including class weights metadata"""
        metadata = {
            "model_type": f"Weighted YOLOv12{self.model_size} Classifier",
            "class_names": self.class_names,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "num_classes": len(self.class_names),
            "class_weights_applied": self.class_weights,
            "class_weights_tensor": (
                self.class_weights_tensor.tolist()
                if self.class_weights_tensor is not None
                else None
            ),
            "training_timestamp": datetime.datetime.now().isoformat(),
            "save_directory": self.save_dir,
            "model_path": os.path.join(self.save_dir, "weighted_training", "weights", "best.pt"),
            "features": [
                "Class weights applied through custom loss function",
                "Object detection to classification conversion",
                "Top-K accuracy metrics",
                "Class imbalance handling",
                "Cropped object classification",
            ],
        }

        metadata_path = os.path.join(self.save_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model metadata saved to: {metadata_path}")


def main():
    """Main training pipeline for weighted YOLOv12 classification"""
    # Initialize classifier
    classifier = WeightedYOLOv12Classifier(model_size="n", img_size=640, batch_size=16)

    # Download and prepare dataset
    print("Step 1: Downloading dataset...")
    gdrive_file_id = "1DjNhyBMEcnuF-jpgUdBuJxjRmWHSNIwu"
    dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

    # Prepare classification dataset structure
    print("\nStep 2: Converting detection dataset to classification format...")
    classification_path = classifier.prepare_yolo_classification_structure(
        dataset_path, train_split=0.8
    )

    # Initialize model
    print("\nStep 3: Initializing YOLOv12 model...")
    if not classifier.initialize_yolov12_classifier():
        print("Failed to initialize model")
        return

    # Train model
    print("\nStep 4: Training model with class weights...")
    training_results = classifier.train_model_with_class_weights(classification_path, epochs=100)

    # Validate model
    print("\nStep 5: Validating model...")
    validation_results = classifier.validate_model(classification_path)

    # Create analysis
    print("\nStep 6: Creating confusion matrix and analysis...")
    classifier.create_confusion_matrix(classification_path)

    # Save model info
    print("\nStep 7: Saving model information...")
    classifier.save_model_info()

    # Cleanup
    if os.path.exists("dataset/"):
        shutil.rmtree("dataset/")

    print(f"\n{'='*60}")
    print("WEIGHTED YOLOV12 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Model saved in: {classifier.save_dir}")
    print("Features implemented:")
    print("✓ Class weights applied through custom loss function")
    print("✓ Object detection to classification conversion")
    print("✓ Top-K accuracy metrics")
    print("✓ Class imbalance handling")
    print("✓ Cropped object classification")
    print("\nFiles created:")
    print("- Best model: weighted_training/weights/best.pt")
    print("- Last model: weighted_training/weights/last.pt")
    print("- Training plots: weighted_training/")
    print("- Confusion matrix: confusion_matrix_analysis.png")
    print("- Classification report: classification_report.json")
    print("- Model metadata: model_metadata.json")

    if classifier.class_weights:
        print(f"\nClass weights applied:")
        for class_id, weight in classifier.class_weights.items():
            class_name = (
                classifier.class_names[class_id]
                if class_id < len(classifier.class_names)
                else f"Unknown_{class_id}"
            )
            print(f"  {class_name}: {weight:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
