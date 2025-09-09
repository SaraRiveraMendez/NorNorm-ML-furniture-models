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


class WeightedYOLOv12Classifier:
    def __init__(self, model_size="n", img_size=640, batch_size=16):
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.class_weights = None
        self.class_weights_tensor = None
        self.progressive_unfreeze_schedule = []

        # Create folder with timestamp for this training session
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/WeightedYOLOv12_Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

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
        """Extract ALL elements from images and split into train/validation 80/20 with class weights"""
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])
                print(f"Classes found: {self.class_names}")

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

        # Create tensor for PyTorch loss functions
        self.class_weights_tensor = torch.ones(len(self.class_names))
        for class_id, weight in self.class_weights.items():
            self.class_weights_tensor[class_id] = weight

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

    def create_weighted_dataset_yaml(self, dataset_path):
        """Create YOLO classification configuration with class weights"""
        # Convert class weights tensor to list for YAML serialization
        class_weights_list = self.class_weights_tensor.tolist()

        config = {
            "path": os.path.abspath(dataset_path),
            "train": "train",
            "val": "valid",
            "names": {i: name for i, name in enumerate(self.class_names)},
            "class_weights": class_weights_list,  # Add class weights to config
            "nc": len(self.class_names),
        }

        config_path = os.path.join(dataset_path, "dataset.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"YOLO classification config with class weights saved to: {config_path}")
        print(f"Class weights included: {[f'{w:.3f}' for w in class_weights_list]}")
        return config_path

    def apply_class_weights_to_model(self):
        """Apply class weights to model's loss function"""
        if self.model is None or self.class_weights_tensor is None:
            print("Model or class weights not available")
            return False

        try:
            # Access the model's loss function
            if hasattr(self.model.model, "loss"):
                loss_fn = self.model.model.loss

                # Apply weights to classification loss
                if hasattr(loss_fn, "cls"):
                    # For classification models, set class weights
                    device = next(self.model.model.parameters()).device
                    loss_fn.cls_pos_weight = self.class_weights_tensor.to(device)
                    print(f"Class weights applied to model loss function on device: {device}")
                    return True

            # Alternative approach: Modify the model's criterion directly
            if hasattr(self.model.model, "criterion"):
                criterion = self.model.model.criterion
                if hasattr(criterion, "weight"):
                    device = next(self.model.model.parameters()).device
                    criterion.weight = self.class_weights_tensor.to(device)
                    print(f"Class weights applied to model criterion on device: {device}")
                    return True

            print("Could not find appropriate loss function to apply class weights")
            return False

        except Exception as e:
            print(f"Error applying class weights to model: {e}")
            return False

    def create_custom_weighted_loss(self):
        """Create custom weighted loss function as backup"""

        class WeightedCrossEntropyLoss(nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.weights = weights
                self.ce_loss = nn.CrossEntropyLoss(weight=weights, reduction="mean")

            def forward(self, predictions, targets):
                return self.ce_loss(predictions, targets)

        if self.class_weights_tensor is not None:
            device = (
                next(self.model.model.parameters()).device if self.model else torch.device("cpu")
            )
            self.weighted_loss = WeightedCrossEntropyLoss(self.class_weights_tensor.to(device))
            print("Custom weighted loss function created")
            return self.weighted_loss
        return None

    def initialize_yolov12_classifier(self):
        """Initialize YOLOv12 model for classification with class weights"""
        try:
            model_name = f"yolo12{self.model_size}-cls.pt"
            self.model = YOLO(model_name)
            print(f"YOLOv12{self.model_size} Classification model initialized successfully")

            # Apply class weights to the model
            if self.class_weights_tensor is not None:
                weight_applied = self.apply_class_weights_to_model()
                if not weight_applied:
                    print("Warning: Could not apply class weights directly to model")
                    self.create_custom_weighted_loss()

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

                # Apply class weights to the model
                if self.class_weights_tensor is not None:
                    weight_applied = self.apply_class_weights_to_model()
                    if not weight_applied:
                        print("Warning: Could not apply class weights directly to model")
                        self.create_custom_weighted_loss()

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

    def train_model_with_class_weights(self, dataset_path, epochs=100):
        """Train YOLOv12 classification model with applied class weights"""
        print(f"\n{'='*60}")
        print("STARTING WEIGHTED YOLOv12 CLASSIFICATION TRAINING")
        print(f"{'='*60}")
        print(f"Model: YOLOv12{self.model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Dataset path: {dataset_path}")

        if self.class_weights:
            print("Class weights APPLIED during training:")
            for class_id, weight in self.class_weights.items():
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"Unknown_{class_id}"
                )
                print(f"  {class_name}: {weight:.3f}")

        print(f"{'='*60}")

        # Create YOLO config with class weights
        config_path = self.create_weighted_dataset_yaml(dataset_path)

        # Train the model with enhanced parameters for weighted training
        training_args = {
            "data": config_path,
            "epochs": epochs,
            "imgsz": self.img_size,
            "batch": self.batch_size,
            "device": "cpu",  # Change to "0" for GPU
            "workers": 8,
            "patience": 25,
            "save": True,
            "save_period": 10,
            "val": True,
            "project": self.save_dir,
            "name": "weighted_training",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 5e-4,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,
            "cls": 1.0,  # Classification loss weight
            "dfl": 1.5,  # Distribution focal loss weight
            "verbose": True,
        }

        # Add class weights if available in the training arguments
        if hasattr(self.model, "args"):
            self.model.args.cls_pos_weight = self.class_weights_tensor

        results = self.model.train(**training_args)

        print("Weighted training completed!")
        print(f"Class weights were applied to handle class imbalance")
        return results

    def validate_model_with_weights(self, dataset_path):
        """Validate the trained model with class weight considerations"""
        print("Validating weighted model...")

        # Use the best trained weights
        best_model_path = os.path.join(self.save_dir, "weighted_training", "weights", "best.pt")
        if os.path.exists(best_model_path):
            self.model = YOLO(best_model_path)
            print(f"Loaded best weighted model from: {best_model_path}")

        # Standard validation
        results = self.model.val()

        # Calculate weighted Top-K accuracies
        top_k_metrics = self.calculate_weighted_top_k_accuracy(dataset_path, k_values=[1, 3])

        # Combine results
        validation_results = {
            "standard_metrics": results,
            "top_k_metrics": top_k_metrics,
            "class_weights_applied": True,
            "class_weights": self.class_weights,
        }

        print("Weighted validation completed!")
        if "top_1_accuracy" in top_k_metrics:
            print(f"Weighted Top-1 Accuracy: {top_k_metrics['top_1_accuracy']:.4f}")
        if "top_3_accuracy" in top_k_metrics:
            print(f"Weighted Top-3 Accuracy: {top_k_metrics['top_3_accuracy']:.4f}")

        return validation_results

    def calculate_weighted_top_k_accuracy(self, dataset_path, k_values=[1, 3]):
        """Calculate Top-K accuracy metrics with class weight consideration"""
        print(f"Calculating Weighted Top-K accuracy for k={k_values}...")

        val_path = os.path.join(dataset_path, "val")
        y_true = []
        y_pred_probs = []
        sample_weights = []

        # Get predictions for all validation images
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(val_path, class_name)
            if os.path.exists(class_path):
                class_weight = self.class_weights.get(class_idx, 1.0)

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
                                sample_weights.append(class_weight)

        if len(y_true) == 0:
            print("No predictions found for weighted accuracy calculation")
            return {}

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        sample_weights = np.array(sample_weights)

        # Calculate Top-K accuracies
        top_k_accuracies = {}
        for k in k_values:
            if k <= len(self.class_names):
                # Standard top-k accuracy
                acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)

                # Weighted top-k accuracy
                weighted_acc = top_k_accuracy_score(
                    y_true, y_pred_probs, k=k, sample_weight=sample_weights
                )

                top_k_accuracies[f"top_{k}_accuracy"] = acc
                top_k_accuracies[f"weighted_top_{k}_accuracy"] = weighted_acc

                print(f"Top-{k} Accuracy: {acc:.4f}")
                print(f"Weighted Top-{k} Accuracy: {weighted_acc:.4f}")

        return top_k_accuracies

    def create_weighted_confusion_matrix(self, dataset_path):
        """Create confusion matrix with class weight considerations"""
        print("Creating weighted confusion matrix...")

        val_path = os.path.join(dataset_path, "valid")
        y_true = []
        y_pred = []
        y_pred_probs = []
        sample_weights = []

        # Get predictions for all validation images
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(val_path, class_name)
            if os.path.exists(class_path):
                class_weight = self.class_weights.get(class_idx, 1.0)

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
                                sample_weights.append(class_weight)

                                # Store probabilities for Top-K calculation
                                if hasattr(probs, "data"):
                                    prob_scores = probs.data.cpu().numpy()
                                    y_pred_probs.append(prob_scores)

        if len(y_true) == 0:
            print("No predictions found for weighted confusion matrix")
            return

        # Calculate weighted Top-K accuracies
        top_k_metrics = self.calculate_weighted_top_k_accuracy(dataset_path, k_values=[1, 3])

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create weighted confusion matrix
        cm_weighted = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)

        # Calculate per-class metrics
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_support = cm.sum(axis=1)

        # Plot side by side comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Standard Confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Standard Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted Labels")
        axes[0, 0].set_ylabel("True Labels")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Weighted Confusion matrix
        sns.heatmap(
            cm_weighted,
            annot=True,
            fmt=".1f",
            cmap="Oranges",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Weighted Confusion Matrix")
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

        # Add Top-K accuracy information
        if top_k_metrics:
            info_text = "Weighted Top-K Accuracies:\n"
            for k, acc in top_k_metrics.items():
                if "weighted" in k:
                    info_text += f"{k.replace('_', '-').title()}: {acc:.4f}\n"
            axes[0, 0].text(
                0.02,
                0.98,
                info_text,
                transform=axes[0, 0].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "weighted_confusion_matrix_yolov12.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Enhanced classification report with class weights
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        report["class_weights_applied"] = self.class_weights
        report["top_k_accuracies"] = top_k_metrics
        report["progressive_unfreeze_schedule"] = self.progressive_unfreeze_schedule
        report["weighted_training"] = True

        report_path = os.path.join(self.save_dir, "weighted_classification_report_yolov12.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Weighted confusion matrix saved to: {save_path}")
        print(f"Weighted classification report saved to: {report_path}")

        # Print weighted Top-K accuracies
        if top_k_metrics:
            print("\nWeighted Top-K Accuracies:")
            for k, acc in top_k_metrics.items():
                if "weighted" in k:
                    print(f"  {k.replace('_', '-').title()}: {acc:.4f}")

    def save_weighted_model_info(self):
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
            "progressive_unfreeze_schedule": self.progressive_unfreeze_schedule,
            "training_timestamp": datetime.datetime.now().isoformat(),
            "save_directory": self.save_dir,
            "model_path": os.path.join(self.save_dir, "weighted_training", "weights", "best.pt"),
            "features": [
                "Class weights applied during training",
                "Progressive unfreezing with weighted loss",
                "Weighted Top-1 and Top-3 accuracy metrics",
                "Class imbalance handling",
                "No image cropping (direct classification format)",
            ],
        }

        metadata_path = os.path.join(self.save_dir, "weighted_model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Weighted model metadata saved to: {metadata_path}")


def main():
    """Main training pipeline for weighted YOLOv12 classification"""
    # Initialize classifier
    classifier = WeightedYOLOv12Classifier(model_size="n", img_size=640, batch_size=16)

    # Download and prepare dataset
    print("Step 1: Downloading dataset...")
    gdrive_file_id = "1nGK6c3TQWzTfI5KpekIm10NP5W2hAswE"
    dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

    # Prepare classification dataset with class weights calculation
    print("\nStep 2: Preparing classification dataset with class weight calculation...")
    (train_images, train_labels), (val_images, val_labels) = (
        classifier.prepare_classification_dataset(dataset_path, train_split=0.8)
    )

    # Initialize model with class weights
    print("\nStep 3: Initializing YOLOv12 model with class weights...")
    if not classifier.initialize_yolov12_classifier():
        print("Failed to initialize model")
        return

    # Train model with applied class weights
    print("\nStep 4: Training model with class weights and progressive unfreezing...")
    training_results = classifier.train_model_with_class_weights(dataset_path, epochs=100)

    # Validate model with weighted metrics
    print("\nStep 5: Validating model with weighted Top-K accuracy...")
    validation_results = classifier.validate_model_with_weights(dataset_path)

    # Create weighted confusion matrix
    print("\nStep 6: Creating weighted confusion matrix with class weight analysis...")
    classifier.create_weighted_confusion_matrix(dataset_path)

    # Save weighted model info
    print("\nStep 7: Saving weighted model information...")
    classifier.save_weighted_model_info()

    # Cleanup temporary files
    if os.path.exists("dataset/") and dataset_path != "dataset/":
        shutil.rmtree("dataset/")

    print(f"\n{'='*60}")
    print("WEIGHTED YOLOV12 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Model saved in: {classifier.save_dir}")
    print("Features implemented:")
    print("✓ Class weights APPLIED during training")
    print("✓ Progressive unfreezing with weighted loss")
    print("✓ Weighted Top-1 and Top-3 accuracy metrics")
    print("✓ Class imbalance handling")
    print("✓ Weighted confusion matrix analysis")
    print("✓ No image cropping (direct classification)")
    print("\nFiles created:")
    print("- Best weighted model: weighted_training/weights/best.pt")
    print("- Last weighted model: weighted_training/weights/last.pt")
    print("- Training plots: weighted_training/")
    print("- Weighted confusion matrix: weighted_confusion_matrix_yolov12.png")
    print("- Weighted classification report: weighted_classification_report_yolov12.json")
    print("- Weighted model metadata: weighted_model_metadata.json")

    # Print class weights summary
    print(f"\nClass weights applied:")
    if classifier.class_weights:
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
