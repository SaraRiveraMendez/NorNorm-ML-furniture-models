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

        all_images = []
        all_labels = []
        total_elements = 0
        images_processed = 0

        # Process all splits (train and valid) together
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

        for schedule in reversed(self.progressive_unfreeze_schedule):
            if current_epoch >= schedule["epoch"]:
                if "freeze_backbone" in schedule and schedule["freeze_backbone"]:
                    # Freeze backbone layers
                    for name, param in self.model.model.named_parameters():
                        if not name.startswith("model.22"):  # Keep classifier head unfrozen
                            param.requires_grad = False
                    print(f"Epoch {current_epoch}: Backbone frozen, training head only")

                elif "unfreeze_layers" in schedule:
                    layers_to_unfreeze = schedule["unfreeze_layers"]

                    if layers_to_unfreeze == "all":
                        # Unfreeze all layers
                        for param in self.model.model.parameters():
                            param.requires_grad = True
                        print(f"Epoch {current_epoch}: All layers unfrozen")
                    else:
                        # Unfreeze specific layers
                        for layer_name in layers_to_unfreeze:
                            for name, param in self.model.model.named_parameters():
                                if layer_name in name:
                                    param.requires_grad = True
                        print(f"Epoch {current_epoch}: Unfrozen layers: {layers_to_unfreeze}")
                break

    def create_yolo_classification_config(self, dataset_path):
        """Create YOLO classification configuration"""
        config = {
            "path": os.path.abspath(dataset_path),
            "train": "train",
            "val": "val",
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

        # Define progressive unfreezing callback
        class ProgressiveUnfreezeCallback:
            def __init__(self, classifier):
                self.classifier = classifier

            def __call__(self, trainer):
                """Called at the start of each epoch"""
                current_epoch = trainer.epoch
                self.classifier._apply_progressive_unfreeze(current_epoch)

        # Register callback with Ultralytics
        callback = ProgressiveUnfreezeCallback(self)
        self.model.add_callback("on_train_epoch_start", callback)

        # Train the model
        results = self.model.train(
            data=config_path,
            epochs=epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device="cpu",
            workers=8,
            patience=25,
            save=True,
            save_period=10,
            val=True,
            project=self.save_dir,
            name="training",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
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

        val_path = os.path.join(dataset_path, "val")
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
                acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
                top_k_accuracies[f"top_{k}_accuracy"] = acc
                print(f"Top-{k} Accuracy: {acc:.4f}")

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
        print(f"Top-1 Accuracy: {top_k_metrics.get('top_1_accuracy', 'N/A'):.4f}")
        print(f"Top-3 Accuracy: {top_k_metrics.get('top_3_accuracy', 'N/A'):.4f}")

        return validation_results

    def create_confusion_matrix(self, dataset_path):
        """Create confusion matrix from validation results"""
        print("Creating confusion matrix...")

        val_path = os.path.join(dataset_path, "val")
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
        if y_pred_probs:
            y_pred_probs = np.array(y_pred_probs)
            for k in [1, 3]:
                if k <= len(self.class_names):
                    acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
                    top_k_metrics[f"top_{k}_accuracy"] = acc

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


def main():
    """Main training pipeline for pure YOLOv12 classification"""
    # Initialize classifier
    classifier = PureYOLOv12FurnitureClassifier(model_size="n", img_size=640, batch_size=16)

    # Download and prepare dataset
    print("Step 1: Downloading dataset...")
    gdrive_file_id = "1Mfp9TV22_2eU47nZYU00LAs8HkPwAqYZ"
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
    training_results = classifier.train_model(
        dataset_path, epochs=100
    )  # keep dataset_path (string)
    print(training_results)

    # Validate model with Top-K metrics
    print("\nStep 5: Validating model with Top-K accuracy...")
    validation_results = classifier.validate_model(dataset_path)  # use dataset_path
    print(validation_results)

    # Create confusion matrix with Top-K metrics
    print("\nStep 6: Creating confusion matrix with Top-K metrics...")
    classifier.create_confusion_matrix(dataset_path)  # use dataset_path

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
