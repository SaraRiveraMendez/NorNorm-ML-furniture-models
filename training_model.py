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


class ImprovedYOLOv12Classifier:
    def __init__(self, model_size="n", img_size=640, batch_size=8):
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.class_weights = None

        # Create folder with timestamp
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/ImprovedYOLOv12_Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def download_and_extract_dataset(self, gdrive_file_id):
        """Download and extract dataset from Google Drive"""
        print("Downloading dataset from Google Drive...")

        # Create download directory
        download_dir = os.path.join(self.save_dir, "downloads")
        os.makedirs(download_dir, exist_ok=True)

        # Download file
        zip_path = os.path.join(download_dir, "dataset.zip")
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        try:
            gdown.download(url, zip_path, quiet=False)
            print(f"Dataset downloaded: {zip_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please manually download the dataset and place it in the downloads folder")
            return None

        # Extract dataset
        extract_path = os.path.join(download_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Dataset extracted to: {extract_path}")

            # Find the actual dataset folder (sometimes it's nested)
            for root, dirs, files in os.walk(extract_path):
                if "data.yaml" in files or any(d in ["train"] for d in dirs):
                    print(f"Found dataset at: {root}")
                    return root

            # If no specific structure found, return the extract path
            return extract_path

        except Exception as e:
            print(f"Extraction failed: {e}")
            return None

    def create_classification_labels(self, classification_dir):
        """Create YOLO-compatible label files for classification dataset"""
        print("Creating YOLO classification labels...")

        for split in ["train", "val"]:
            split_dir = os.path.join(classification_dir, split)
            labels_dir = os.path.join(classification_dir, split, "labels")
            os.makedirs(labels_dir, exist_ok=True)

            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.exists(class_dir):
                    continue

                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        # Create corresponding label file
                        label_filename = os.path.splitext(img_file)[0] + ".txt"
                        label_path = os.path.join(labels_dir, label_filename)

                        # Write class index to label file
                        with open(label_path, "w") as f:
                            f.write(str(class_idx))

        print("Label files created successfully")

    def clean_and_extract_objects(self, dataset_path, min_area=0.0001, max_samples_per_class=10000):
        """Clean YOLO detection dataset (remove background, invalid bboxes) and keep YOLO format."""
        print("Extracting and cleaning detection annotations...")

        # Load class names
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"{yaml_path} not found")

        with open(yaml_path, "r") as f:
            data_config = yaml.safe_load(f)
            self.class_names = data_config.get("names", [])

        # Normalize & remove background
        original_names = [n.strip().lower() for n in self.class_names]
        self.class_names = [n for n in original_names if n not in ["background", "bg"]]

        if len(self.class_names) < len(original_names):
            print(f"Removed background class(es). {len(original_names)} → {len(self.class_names)}")

        print(f"Target classes: {self.class_names}")

        # Map old indices → new indices
        id_map = {
            i: self.class_names.index(n)
            for i, n in enumerate(original_names)
            if n in self.class_names
        }

        # Prepare cleaned dataset dir
        classification_dir = os.path.join(self.save_dir, "cleaned_dataset")
        os.makedirs(classification_dir, exist_ok=True)

        # Save new yaml
        new_yaml = {
            "train": os.path.join(classification_dir, "train"),
            "val": os.path.join(classification_dir, "val"),
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        with open(os.path.join(classification_dir, "data.yaml"), "w") as f:
            yaml.safe_dump(new_yaml, f)

        # Track counts
        class_counts = {name: 0 for name in self.class_names}
        total_labels, kept_labels = 0, 0

        # Process splits
        for split in ["train", "val", "valid"]:
            images_dir = os.path.join(dataset_path, split, "images")
            labels_dir = os.path.join(dataset_path, split, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue

            print(f"Processing {split} split...")

            new_images_dir = os.path.join(classification_dir, split, "images")
            new_labels_dir = os.path.join(classification_dir, split, "labels")
            os.makedirs(new_images_dir, exist_ok=True)
            os.makedirs(new_labels_dir, exist_ok=True)

            for img_file in os.listdir(images_dir):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(images_dir, img_file)
                new_img_path = os.path.join(new_images_dir, img_file)

                # Copy image
                shutil.copy2(img_path, new_img_path)

                # Process label
                label_file = os.path.splitext(img_file)[0] + ".txt"
                old_label_path = os.path.join(labels_dir, label_file)
                new_label_path = os.path.join(new_labels_dir, label_file)

                if not os.path.exists(old_label_path):
                    continue

                with open(old_label_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    if class_id not in id_map:
                        continue

                    cx, cy, bw, bh = map(float, parts[1:])

                    # Skip invalid
                    if bw <= 0 or bh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
                        continue
                    if bw * bh < min_area:
                        continue

                    new_id = id_map[class_id]
                    new_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    class_counts[self.class_names[new_id]] += 1
                    kept_labels += 1

                total_labels += len(lines)

                if new_lines:
                    with open(new_label_path, "w") as f:
                        f.writelines(new_lines)

        print(f"\nObject cleaning summary:")
        print(f"Original labels: {total_labels}")
        print(f"Kept labels: {kept_labels}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} objects")

        return os.path.join(classification_dir, "data.yaml")

    def create_yolo_classification_config(self, classification_dir):
        """Create YOLO classification configuration"""
        config = {
            "path": os.path.abspath(classification_dir),
            "train": "train",
            "val": "val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        config_path = os.path.join(classification_dir, "dataset.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"YOLO classification config saved: {config_path}")
        print(f"Number of classes: {len(self.class_names)}")

        return config_path

    def initialize_model(self):
        """Initialize YOLOv12 classification model"""
        try:
            # Try classification model first
            model_name = f"yolo12{self.model_size}-cls.pt"
            self.model = YOLO(model_name)
            print(f"YOLOv12{self.model_size} classification model loaded")
            return True
        except:
            try:
                # Fallback to detection model
                model_name = f"yolo12{self.model_size}.pt"
                self.model = YOLO(model_name)
                print(
                    f"YOLOv12{self.model_size} detection model loaded (will adapt for classification)"
                )
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False

    def train_with_progressive_unfreezing(self, config_path, total_epochs=80):
        """Entrenamiento con descongelamiento progresivo"""
        print(f"Starting progressive unfreezing training...")
        print(f"Classes: {len(self.class_names)}")
        print(f"Total epochs: {total_epochs}")

        # Definir las fases de descongelamiento
        phases = [
            {
                "name": "Phase 1: Frozen Backbone",
                "epochs": total_epochs // 3,
                "freeze_backbone": True,
                "freeze_neck": False,
                "lr": 0.001,
                "description": "Only training the head",
            },
            {
                "name": "Phase 2: Partial Unfreezing",
                "epochs": total_epochs // 3,
                "freeze_backbone": False,
                "freeze_neck": True,
                "lr": 0.0005,
                "description": "Backbone unfrezing, the neck is still frozen",
            },
            {
                "name": "Phase 3: Full Unfreezing",
                "epochs": total_epochs - 2 * (total_epochs // 3),
                "freeze_backbone": False,
                "freeze_neck": False,
                "lr": 0.0001,
                "description": "Train all the model with a low LR",
            },
        ]

        print(f"\n{'='*60}")
        print("PROGRESSIVE UNFREEZING TRAINING PLAN:")
        for i, phase in enumerate(phases):
            print(f"{phase['name']}: {phase['epochs']} epochs, LR: {phase['lr']}")
            print(f"  - {phase['description']}")
        print(f"{'='*60}\n")

        try:
            current_epoch = 0

            for phase_idx, phase in enumerate(phases):
                print(f"\nStarting {phase['name']}")
                print(f"Epochs: {phase['epochs']}, Learning Rate: {phase['lr']}")
                print("-" * 50)

                # Training parameters for this phase
                train_kwargs = {
                    "data": config_path,
                    "epochs": phase["epochs"],
                    "imgsz": self.img_size,
                    "batch": self.batch_size,
                    "device": "cpu",  # Change to "0" if you have GPU
                    "workers": 2,
                    "patience": max(10, phase["epochs"] // 3),
                    "save": True,
                    "save_period": max(5, phase["epochs"] // 4),
                    "val": True,
                    "project": self.save_dir,
                    "name": f"phase_{phase_idx + 1}_training",
                    "exist_ok": True,
                    "pretrained": True,
                    "resume": phase_idx > 0,
                    "optimizer": "AdamW",
                    "lr0": phase["lr"],
                    "lrf": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 0.001,
                    "warmup_epochs": min(3, phase["epochs"] // 5),
                    "warmup_momentum": 0.8,
                    "warmup_bias_lr": phase["lr"] * 0.1,
                    "hsv_h": 0.01 if phase_idx == 0 else 0.02,
                    "hsv_s": 0.5 if phase_idx == 0 else 0.7,
                    "hsv_v": 0.3 if phase_idx == 0 else 0.4,
                    "degrees": 5 if phase_idx == 0 else 10,
                    "translate": 0.1 if phase_idx == 0 else 0.2,
                    "scale": 0.3 if phase_idx == 0 else 0.5,
                    "flipud": 0.3 if phase_idx == 0 else 0.5,
                    "fliplr": 0.5,
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "cls": 1.0,
                    "box": 0.0,
                    "dfl": 0.0,
                    "verbose": True,
                    "plots": True,
                }

                # Freezing configuration for this phase:
                if phase_idx == 0:
                    # Phase 1: Backbone freezing
                    train_kwargs["freeze"] = 10  # Freeze first 10 layers
                elif phase_idx == 1:
                    # Phase 2: Ufreezing the backbone partially
                    train_kwargs["freeze"] = 5  # Freeze first 5 layers
                else:
                    # Phase 3: Unfreeze everything
                    train_kwargs["freeze"] = 0  # No freezing

                phase_results = self.model.train(**train_kwargs)

                current_epoch += phase["epochs"]
                print(f"Completed {phase['name']}")
                print(f"Total epochs completed: {current_epoch}/{total_epochs}")

                print(f"\nEvaluating {phase['name']}...")
                try:
                    val_results = self.model.val()
                    print(f"Phase {phase_idx + 1} validation completed")
                    if hasattr(val_results, "top1"):
                        print(f"Phase {phase_idx + 1} Top-1 Accuracy: {val_results.top1:.3f}")
                except Exception as e:
                    print(f"Phase {phase_idx + 1} validation error: {e}")

            print(f"\nProgressive unfreezing training completed!")
            return phase_results

        except Exception as e:
            print(f"Progressive training failed: {e}")
            print("Trying fallback standard training...")
            return self.train_with_improved_params(config_path, total_epochs)

    def train_with_improved_params(self, config_path, epochs=80):
        """Train with optimized parameters for better accuracy (fallback method)"""
        print(f"Starting fallback training with improved parameters...")

        try:
            # Enhanced training parameters
            results = self.model.train(
                data=config_path,
                epochs=epochs,
                imgsz=self.img_size,
                batch=self.batch_size,
                device="cpu",
                workers=2,
                patience=15,
                save=True,
                save_period=10,
                val=True,
                project=self.save_dir,
                name="training_fallback",
                exist_ok=True,
                pretrained=True,
                optimizer="AdamW",
                lr0=0.001,
                lrf=0.1,
                momentum=0.9,
                weight_decay=0.001,
                warmup_epochs=5,
                warmup_momentum=0.8,
                warmup_bias_lr=0.01,
                hsv_h=0.02,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10,
                translate=0.2,
                scale=0.5,
                flipud=0.5,
                fliplr=0.5,
                mosaic=0.0,
                mixup=0.0,
                cls=1.0,
                box=0.0,
                dfl=0.0,
                verbose=True,
                plots=True,
            )

            print("Fallback training completed successfully!")
            return results

        except Exception as e:
            print(f"Fallback training also failed: {e}")
            return None

    def evaluate_model(self, classification_dir):
        """Comprehensive evaluation"""
        print("Evaluating trained model...")

        # Load best model
        best_model_path = self.find_best_model()
        if best_model_path:
            self.model = YOLO(best_model_path)
            print(f"Loaded best model from: {best_model_path}")

        # Standard validation
        try:
            results = self.model.val()
            if hasattr(results, "top1"):
                print(f"Validation Top-1 Accuracy: {results.top1:.3f}")
        except Exception as e:
            print(f"Validation error: {e}")

        # Manual evaluation on validation set
        val_dir = os.path.join(classification_dir, "val")
        y_true = []
        y_pred = []
        y_pred_probs = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(val_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, img_file)

                    try:
                        # Predict
                        pred_results = self.model.predict(img_path, verbose=False)

                        if pred_results and len(pred_results) > 0:
                            result = pred_results[0]

                            if hasattr(result, "probs") and result.probs is not None:
                                # Classification result
                                predicted_class = result.probs.top1
                                confidence = result.probs.top1conf

                                y_true.append(class_idx)
                                y_pred.append(predicted_class)

                                # Get all probabilities for top-k calculation
                                if hasattr(result.probs, "data"):
                                    probs = result.probs.data.cpu().numpy()
                                    y_pred_probs.append(probs)

                    except Exception as e:
                        print(f"Error predicting {img_path}: {e}")
                        continue

        if len(y_true) == 0:
            print("No predictions available for evaluation")
            return None

        # Calculate metrics
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"Manual Evaluation Accuracy: {accuracy:.4f}")

        # Create confusion matrix
        self.create_confusion_matrix(y_true, y_pred)

        return accuracy

    def find_best_model(self):
        """Find the best model from training phases"""
        possible_paths = [
            os.path.join(self.save_dir, "phase_3_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "phase_2_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "phase_1_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "training_fallback", "weights", "best.pt"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def create_confusion_matrix(self, y_true, y_pred):
        """Create detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        # Calculate per-class metrics
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

        # Create visualization
        plt.figure(figsize=(12, 10))

        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show actual counts
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Normalized Frequency"},
        )

        plt.title("Progressive Unfreezing YOLOv12 Classifier - Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Add accuracy text
        for i in range(len(self.class_names)):
            plt.text(
                len(self.class_names) + 0.5,
                i + 0.5,
                f"Acc: {per_class_accuracy[i]:.3f}",
                ha="center",
                va="center",
            )

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, "progressive_unfreezing_confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Confusion matrix saved: {save_path}")

        # Save detailed report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        with open(os.path.join(self.save_dir, "progressive_unfreezing_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Print per-class performance
        print("\nPer-class Performance:")
        for i, class_name in enumerate(self.class_names):
            if i < len(per_class_accuracy):
                support = cm.sum(axis=1)[i]
                print(f"  {class_name}: {per_class_accuracy[i]:.3f} accuracy ({support} samples)")


def main():
    """Improved training pipeline with progressive unfreezing"""
    print("Starting YOLOv12 Classification Training with Progressive Unfreezing")
    print("=" * 70)

    # Initialize classifier
    classifier = ImprovedYOLOv12Classifier(model_size="n", img_size=640, batch_size=8)

    try:
        # Download dataset
        print("Step 1: Downloading dataset...")
        gdrive_file_id = "1M0e7oXsqs9BQRKxzBzMXKSUSg9JmHfzn"
        dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

        # Extract and clean objects
        print("\nStep 2: Extracting objects and removing background...")
        classification_dir = classifier.clean_and_extract_objects(
            dataset_path, max_samples_per_class=10000
        )

        # Create YOLO config
        print("\nStep 3: Creating YOLO classification config...")
        config_path = classifier.create_yolo_classification_config(classification_dir)

        # Initialize model
        print("\nStep 4: Initializing YOLOv12 model...")
        if not classifier.initialize_model():
            print("Failed to initialize model")
            return

        # Train with progressive unfreezing
        print("\nStep 5: Training with progressive unfreezing...")
        results = classifier.train_with_progressive_unfreezing(config_path, total_epochs=80)

        # Evaluate
        print("\nStep 6: Comprehensive evaluation...")
        final_accuracy = classifier.evaluate_model(classification_dir)

        print(f"\n{'='*70}")
        print("PROGRESSIVE UNFREEZING TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Results saved in: {classifier.save_dir}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
