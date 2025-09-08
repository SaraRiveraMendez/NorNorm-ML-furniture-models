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
    def __init__(self, model_size="n", img_size=640, batch_size=16):
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
                if "data.yaml" in files or any(d in ["train", "val", "valid"] for d in dirs):
                    print(f"Found dataset at: {root}")
                    return root

            # If no specific structure found, return the extract path
            return extract_path

        except Exception as e:
            print(f"Extraction failed: {e}")
            return None

    def clean_and_extract_objects(self, dataset_path, min_area=0.005, max_samples_per_class=5000):
        """Extract objects from detection data and create proper classification structure"""
        print("Extracting and cleaning objects from detection annotations...")

        # Load class names and remove background if present
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])

        # Filter out background class - this is critical for your accuracy issue
        original_names = self.class_names.copy()
        self.class_names = [
            name for name in self.class_names if name.lower() not in ["background", "bg"]
        ]

        if len(self.class_names) < len(original_names):
            print(
                f"Removed background class(es). Classes: {len(original_names)} -> {len(self.class_names)}"
            )

        print(f"Target furniture classes: {self.class_names}")

        # Create classification directory structure
        classification_dir = os.path.join(self.save_dir, "classification_dataset")
        for split in ["train", "val"]:
            for class_name in self.class_names:
                os.makedirs(os.path.join(classification_dir, split, class_name), exist_ok=True)

        total_extracted = 0
        class_counts = {name: 0 for name in self.class_names}

        # Process all splits
        for split in ["train", "val", "valid"]:
            images_dir = os.path.join(dataset_path, split, "images")
            labels_dir = os.path.join(dataset_path, split, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue

            print(f"Processing {split} split...")

            for img_file in os.listdir(images_dir):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(images_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_file)

                if not os.path.exists(label_path):
                    continue

                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                h, w = image.shape[:2]

                try:
                    with open(label_path, "r") as f:
                        lines = f.readlines()

                    object_count = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        class_id = int(parts[0])

                        # Map to new class indices (without background)
                        if class_id >= len(original_names):
                            continue

                        original_class_name = original_names[class_id]
                        if original_class_name.lower() in ["background", "bg"]:
                            continue  # Skip background

                        if original_class_name not in self.class_names:
                            continue

                        # Check if we have enough samples for this class
                        if class_counts[original_class_name] >= max_samples_per_class:
                            continue

                        # Parse bounding box
                        cx, cy, bw, bh = map(float, parts[1:])

                        # Filter very small objects
                        if bw * bh < min_area:
                            continue

                        # Add padding to bounding box for better context
                        padding = 0.1  # 10% padding
                        bw_padded = min(1.0, bw * (1 + padding))
                        bh_padded = min(1.0, bh * (1 + padding))

                        # Convert to pixel coordinates
                        x1 = int(max(0, (cx - bw_padded / 2) * w))
                        y1 = int(max(0, (cy - bh_padded / 2) * h))
                        x2 = int(min(w, (cx + bw_padded / 2) * w))
                        y2 = int(min(h, (cy + bh_padded / 2) * h))

                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue

                        # Crop and resize object
                        cropped = image[y1:y2, x1:x2]

                        # Resize to fixed size for consistency
                        cropped_resized = cv2.resize(cropped, (224, 224))

                        # Determine target split (80% train, 20% val)
                        target_split = "train" if np.random.random() < 0.8 else "val"

                        # Save cropped object
                        save_filename = f"{os.path.splitext(img_file)[0]}_obj{object_count}.jpg"
                        save_path = os.path.join(
                            classification_dir, target_split, original_class_name, save_filename
                        )

                        cv2.imwrite(save_path, cropped_resized)

                        class_counts[original_class_name] += 1
                        total_extracted += 1
                        object_count += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

        print(f"\nObject extraction summary:")
        print(f"Total objects extracted: {total_extracted}")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} objects")

        # Calculate class distribution and weights
        all_counts = list(class_counts.values())
        if len(all_counts) > 0 and min(all_counts) > 0:
            # Calculate inverse frequency weights
            total_samples = sum(all_counts)
            weights = {}
            for class_name, count in class_counts.items():
                weights[class_name] = total_samples / (len(self.class_names) * count)

            print(f"\nCalculated class weights:")
            for class_name, weight in weights.items():
                print(f"  {class_name}: {weight:.3f}")

        return classification_dir

    def create_yolo_classification_config(self, classification_dir):
        """Create YOLO classification configuration"""
        config = {
            "path": os.path.abspath(classification_dir),
            "train": "train",
            "val": "val",
            "nc": len(self.class_names),
            "names": self.class_names,  # Direct list instead of dict
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

    def freeze_layers(self, model, freeze_backbone=True, freeze_neck=False):
        """Congelar capas específicas del modelo"""
        if not hasattr(model, "model"):
            return

        model_layers = model.model

        # Congelar backbone (primeras capas)
        if freeze_backbone:
            for i, layer in enumerate(model_layers):
                if i < len(model_layers) * 0.7:  # Congela el 70% inicial del modelo
                    for param in layer.parameters():
                        param.requires_grad = False
                    print(f"Frozen layer {i}")

        # Congelar neck (capas intermedias) si se especifica
        if freeze_neck:
            for i, layer in enumerate(model_layers):
                if len(model_layers) * 0.7 <= i < len(model_layers) * 0.9:
                    for param in layer.parameters():
                        param.requires_grad = False
                    print(f"Frozen neck layer {i}")

    def unfreeze_layers(self, model, unfreeze_from_layer=0):
        """Descongelar capas desde una capa específica"""
        if not hasattr(model, "model"):
            return

        model_layers = model.model

        for i, layer in enumerate(model_layers):
            if i >= unfreeze_from_layer:
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"Unfrozen layer {i}")

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
                "description": "Solo entrenar cabezal de clasificación",
            },
            {
                "name": "Phase 2: Partial Unfreezing",
                "epochs": total_epochs // 3,
                "freeze_backbone": False,
                "freeze_neck": True,
                "lr": 0.0005,
                "description": "Descongelar backbone, mantener neck congelado",
            },
            {
                "name": "Phase 3: Full Unfreezing",
                "epochs": total_epochs - 2 * (total_epochs // 3),
                "freeze_backbone": False,
                "freeze_neck": False,
                "lr": 0.0001,
                "description": "Entrenar todo el modelo con LR bajo",
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
                print(f"\n🚀 Starting {phase['name']}")
                print(f"Epochs: {phase['epochs']}, Learning Rate: {phase['lr']}")
                print("-" * 50)

                # Configurar el congelamiento para esta fase
                if phase_idx == 0:
                    # Fase 1: Congelar backbone
                    self.freeze_layers(self.model, freeze_backbone=True, freeze_neck=False)
                elif phase_idx == 1:
                    # Fase 2: Descongelar backbone, congelar neck
                    self.unfreeze_layers(self.model, unfreeze_from_layer=0)
                    self.freeze_layers(self.model, freeze_backbone=False, freeze_neck=True)
                else:
                    # Fase 3: Descongelar todo
                    self.unfreeze_layers(self.model, unfreeze_from_layer=0)

                # Parámetros de entrenamiento para esta fase
                phase_results = self.model.train(
                    data=config_path,
                    epochs=phase["epochs"],
                    imgsz=self.img_size,
                    batch=self.batch_size,
                    device="cpu",  # Change to "0" if you have GPU
                    workers=2,
                    patience=max(10, phase["epochs"] // 3),
                    save=True,
                    save_period=max(5, phase["epochs"] // 4),
                    val=True,
                    project=self.save_dir,
                    name=f"phase_{phase_idx + 1}_training",
                    exist_ok=True,
                    pretrained=True,
                    resume=phase_idx > 0,  # Reanudar desde la fase anterior
                    # Parámetros optimizados por fase
                    optimizer="AdamW",
                    lr0=phase["lr"],
                    lrf=0.1,
                    momentum=0.9,
                    weight_decay=0.001,
                    warmup_epochs=min(3, phase["epochs"] // 5),
                    warmup_momentum=0.8,
                    warmup_bias_lr=phase["lr"] * 0.1,
                    # Data augmentation adaptativa por fase
                    hsv_h=0.01 if phase_idx == 0 else 0.02,
                    hsv_s=0.5 if phase_idx == 0 else 0.7,
                    hsv_v=0.3 if phase_idx == 0 else 0.4,
                    degrees=5 if phase_idx == 0 else 10,
                    translate=0.1 if phase_idx == 0 else 0.2,
                    scale=0.3 if phase_idx == 0 else 0.5,
                    shear=0.0,
                    perspective=0.0,
                    flipud=0.3 if phase_idx == 0 else 0.5,
                    fliplr=0.5,
                    mosaic=0.0,
                    mixup=0.0,
                    # Loss weights
                    cls=1.0,
                    box=0.0,
                    dfl=0.0,
                    verbose=True,
                    plots=True,
                )

                current_epoch += phase["epochs"]
                print(f"Completed {phase['name']}")
                print(f"Total epochs completed: {current_epoch}/{total_epochs}")

                print(f"\nEvaluating {phase['name']}...")
                try:
                    val_results = self.model.val()
                    print(f"Phase {phase_idx + 1} validation completed")
                except Exception as e:
                    print(f"Phase {phase_idx + 1} validation error: {e}")

            print(f"\n🎉 Progressive unfreezing training completed!")
            return phase_results

        except Exception as e:
            print(f"Progressive training failed: {e}")
            print("Trying fallback standard training...")

            # Fallback a entrenamiento estándar
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
                shear=0.0,
                perspective=0.0,
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

        # Load best model from the final phase
        possible_paths = [
            os.path.join(self.save_dir, "phase_3_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "phase_2_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "phase_1_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "training_fallback", "weights", "best.pt"),
        ]

        best_model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                best_model_path = path
                break

        if best_model_path:
            self.model = YOLO(best_model_path)
            print(f"Loaded best model from: {best_model_path}")

        # Standard validation
        results = self.model.val()

        # Detailed evaluation on validation set
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
                                    # Ensure we have the right number of classes
                                    if len(probs) == len(self.class_names):
                                        y_pred_probs.append(probs)
                                    else:
                                        # Pad or truncate to match class count
                                        padded_probs = np.zeros(len(self.class_names))
                                        min_len = min(len(probs), len(self.class_names))
                                        padded_probs[:min_len] = probs[:min_len]
                                        y_pred_probs.append(padded_probs)

                    except Exception as e:
                        print(f"Error predicting {img_path}: {e}")
                        continue

        if len(y_true) == 0:
            print("No predictions available for evaluation")
            return None

        # Calculate metrics
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"Overall Accuracy: {accuracy:.4f}")

        # Top-K accuracies
        if y_pred_probs:
            y_pred_probs = np.array(y_pred_probs)

            try:
                top1_acc = top_k_accuracy_score(y_true, y_pred_probs, k=1)
                print(f"Top-1 Accuracy: {top1_acc:.4f}")
            except:
                print("Could not calculate Top-1 accuracy")

            if len(self.class_names) >= 3:
                try:
                    top3_acc = top_k_accuracy_score(y_true, y_pred_probs, k=3)
                    print(f"Top-3 Accuracy: {top3_acc:.4f}")
                except:
                    print("Could not calculate Top-3 accuracy")

        # Create confusion matrix
        self.create_confusion_matrix(y_true, y_pred)

        return accuracy

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
        # Download dataset (you need to implement this method)
        print("Step 1: Downloading dataset...")
        gdrive_file_id = "1M0e7oXsqs9BQRKxzBzMXKSUSg9JmHfzn"
        dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

        # Extract and clean objects
        print("\nStep 2: Extracting objects and removing background...")
        classification_dir = classifier.clean_and_extract_objects(
            dataset_path, min_area=0.005, max_samples_per_class=3000
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
        print("\nProgressive Unfreezing Applied:")
        print("✓ Phase 1: Frozen backbone (first 1/3 epochs)")
        print("✓ Phase 2: Unfrozen backbone + frozen neck (second 1/3 epochs)")
        print("✓ Phase 3: Fully unfrozen with low LR (final 1/3 epochs)")
        print("✓ Adaptive learning rates per phase")
        print("✓ Progressive data augmentation intensity")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
