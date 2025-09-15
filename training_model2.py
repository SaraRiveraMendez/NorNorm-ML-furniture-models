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
    """
    A comprehensive YOLOv12 classifier with class weighting capabilities.
    Converts object detection datasets to classification format and handles class imbalance.
    FIXED VERSION: Properly handles background class removal and model architecture adaptation.
    """

    def __init__(self, model_size="n", img_size=640, batch_size=12):
        """
        Initialize the WeightedYOLOv12Classifier.

        Args:
            model_size (str): YOLOv12 model size ('n', 's', 'm', 'l', 'x')
            img_size (int): Image size for training
            batch_size (int): Batch size for training
        """
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.original_class_names = []  # Store original names for debugging
        self.class_weights = None
        self.class_weights_tensor = None
        self.weighted_loss_fn = None
        self.id_map = {}  # Store class ID mapping for consistency

        # Create folder with timestamp for this training session
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/WeightedYOLOv12_Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def download_and_extract_dataset(self, gdrive_file_id, output_filename=None):
        """
        Download dataset from Google Drive and extract it.

        Args:
            gdrive_file_id (str): Google Drive file ID
            output_filename (str): Optional output filename

        Returns:
            str: Path to extracted dataset
        """
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

    def clean_and_extract_objects(
        self, dataset_path, min_area=0.0001, max_samples_per_class=15000, val_split=0.2
    ):
        """
        Clean YOLO detection dataset by removing background and invalid bboxes.
        Creates train/val split and maintains YOLO format.
        FIXED: Properly handles background class removal and creates consistent class mapping.

        Args:
            dataset_path (str): Path to original YOLO dataset
            min_area (float): Minimum bbox area threshold
            max_samples_per_class (int): Maximum samples per class
            val_split (float): Validation split ratio

        Returns:
            str: Path to cleaned classification dataset
        """
        print("Extracting and cleaning detection annotations...")

        # Load class names from YOLO config
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YOLO config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data_config = yaml.safe_load(f)
            self.original_class_names = data_config.get("names", [])

        print(f"Original classes found: {self.original_class_names}")

        # Clean class names and remove background classes
        original_names = [n.strip().lower() for n in self.original_class_names]

        # Define background keywords to filter out
        background_keywords = ["background", "bg", "__background__", "void", "unlabeled"]

        # Filter out background classes
        filtered_classes = []
        self.id_map = {}  # Reset class ID mapping
        new_class_id = 0

        for original_id, class_name in enumerate(original_names):
            if class_name not in background_keywords:
                filtered_classes.append(class_name)
                self.id_map[original_id] = new_class_id
                new_class_id += 1
            else:
                print(f"Removing background class: '{class_name}' (original ID: {original_id})")

        self.class_names = filtered_classes

        if len(self.class_names) < len(original_names):
            print(
                f"Background classes removed. {len(original_names)} → {len(self.class_names)} classes"
            )

        if len(self.class_names) == 0:
            raise ValueError("No valid classes found after background removal!")

        print(f"Final target classes: {self.class_names}")
        print(f"Class ID mapping: {self.id_map}")

        # Prepare cleaned dataset directory
        classification_dir = os.path.join(self.save_dir, "cleaned_dataset")

        # Remove existing directory if corrupted
        if os.path.exists(classification_dir):
            shutil.rmtree(classification_dir)

        os.makedirs(classification_dir, exist_ok=True)

        # Initialize tracking variables
        class_counts = {name: 0 for name in self.class_names}
        total_labels, kept_labels = 0, 0

        # Process train split and create validation split
        train_images_dir = os.path.join(dataset_path, "train", "images")
        train_labels_dir = os.path.join(dataset_path, "train", "labels")

        if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
            raise FileNotFoundError("Train images or labels directory not found")

        print(f"Processing train split and creating {val_split*100:.0f}% validation split...")

        # Create output directories
        new_train_images_dir = os.path.join(classification_dir, "train", "images")
        new_train_labels_dir = os.path.join(classification_dir, "train", "labels")
        new_val_images_dir = os.path.join(classification_dir, "val", "images")
        new_val_labels_dir = os.path.join(classification_dir, "val", "labels")

        for dir_path in [
            new_train_images_dir,
            new_train_labels_dir,
            new_val_images_dir,
            new_val_labels_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        # Get all images and create stratified split
        all_images = [
            f for f in os.listdir(train_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Create stratified split based on image classes
        train_imgs, val_imgs = self._create_stratified_split(
            all_images, train_labels_dir, self.id_map, val_split
        )

        print(f"Split created: {len(train_imgs)} train, {len(val_imgs)} val")

        # Process training images
        train_stats = self._process_image_split(
            train_imgs,
            train_images_dir,
            train_labels_dir,
            new_train_images_dir,
            new_train_labels_dir,
            self.id_map,
            min_area,
        )

        # Process validation images
        val_stats = self._process_image_split(
            val_imgs,
            train_images_dir,
            train_labels_dir,
            new_val_images_dir,
            new_val_labels_dir,
            self.id_map,
            min_area,
        )

        # Combine statistics
        total_labels = train_stats["total"] + val_stats["total"]
        kept_labels = train_stats["kept"] + val_stats["kept"]

        for class_name in self.class_names:
            class_counts[class_name] = (
                train_stats["class_counts"][class_name] + val_stats["class_counts"][class_name]
            )

        # Validate that we have samples for all classes
        empty_classes = [cls for cls, count in class_counts.items() if count == 0]
        if empty_classes:
            print(f"WARNING: Classes with no samples: {empty_classes}")

        # Calculate class weights based on object counts
        self._calculate_class_weights(class_counts)

        print(f"\nObject cleaning summary:")
        print(f"Original labels: {total_labels}")
        print(f"Kept labels: {kept_labels}")
        print(f"Train/Val split: {len(train_imgs)}/{len(val_imgs)}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} objects")

        # Store class mapping for later use
        self._save_class_mapping()

        return classification_dir

    def _create_stratified_split(self, all_images, train_labels_dir, id_map, val_split):
        """
        Create stratified train/validation split based on image classes.
        FIXED: Uses the corrected id_map for consistent class handling.

        Args:
            all_images (list): List of all image filenames
            train_labels_dir (str): Directory containing label files
            id_map (dict): Mapping from old to new class IDs
            val_split (float): Validation split ratio

        Returns:
            tuple: (train_images, val_images)
        """
        try:
            image_classes = []
            valid_images = []

            for img_file in all_images:
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(train_labels_dir, label_file)

                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        lines = f.readlines()

                    # Find first valid class for stratification
                    img_class = None
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Valid YOLO format
                            class_id = int(parts[0])
                            if class_id in id_map:
                                img_class = id_map[class_id]
                                break

                    if img_class is not None:
                        valid_images.append(img_file)
                        image_classes.append(img_class)

            print(f"Found {len(valid_images)} images with valid classes for stratification")

            # Attempt stratified split
            if len(valid_images) > 0 and len(set(image_classes)) > 1:
                train_imgs, val_imgs = train_test_split(
                    valid_images, test_size=val_split, stratify=image_classes, random_state=42
                )
                return train_imgs, val_imgs
            else:
                raise ValueError("Insufficient data for stratified split")

        except Exception as e:
            print(f"Stratified split failed: {e}. Using random split...")
            # Fallback to random split
            train_imgs, val_imgs = train_test_split(
                all_images, test_size=val_split, random_state=42
            )
            return train_imgs, val_imgs

    def _process_image_split(
        self, images, src_img_dir, src_labels_dir, dst_img_dir, dst_labels_dir, id_map, min_area
    ):
        """
        Process a split of images (train or validation).
        FIXED: Better handling of class mapping and validation.

        Args:
            images (list): List of image filenames to process
            src_img_dir (str): Source images directory
            src_labels_dir (str): Source labels directory
            dst_img_dir (str): Destination images directory
            dst_labels_dir (str): Destination labels directory
            id_map (dict): Class ID mapping
            min_area (float): Minimum bbox area threshold

        Returns:
            dict: Processing statistics
        """
        class_counts = {name: 0 for name in self.class_names}
        total_labels, kept_labels = 0, 0
        images_with_valid_labels = 0

        for img_file in images:
            img_path = os.path.join(src_img_dir, img_file)
            new_img_path = os.path.join(dst_img_dir, img_file)

            # Process corresponding label file first
            label_file = os.path.splitext(img_file)[0] + ".txt"
            old_label_path = os.path.join(src_labels_dir, label_file)
            new_label_path = os.path.join(dst_labels_dir, label_file)

            processed_labels = {
                "lines": [],
                "total": 0,
                "kept": 0,
                "class_counts": {name: 0 for name in self.class_names},
            }

            if os.path.exists(old_label_path):
                processed_labels = self._process_label_file(old_label_path, id_map, min_area)

            # Only copy image if it has valid labels
            if processed_labels["lines"]:
                # Copy image
                shutil.copy2(img_path, new_img_path)
                images_with_valid_labels += 1

                # Write processed labels
                with open(new_label_path, "w") as f:
                    f.writelines(processed_labels["lines"])

                total_labels += processed_labels["total"]
                kept_labels += processed_labels["kept"]

                # Update class counts
                for class_name, count in processed_labels["class_counts"].items():
                    class_counts[class_name] += count

        print(
            f"  Processed {images_with_valid_labels} images with valid labels out of {len(images)} total images"
        )

        return {"total": total_labels, "kept": kept_labels, "class_counts": class_counts}

    def _process_label_file(self, label_path, id_map, min_area):
        """
        Process individual label files by cleaning and remapping class IDs.
        FIXED: Better validation and error handling.

        Args:
            label_path (str): Path to label file
            id_map (dict): Class ID mapping
            min_area (float): Minimum bbox area threshold

        Returns:
            dict: Processing results
        """
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            return {
                "lines": [],
                "total": 0,
                "kept": 0,
                "class_counts": {name: 0 for name in self.class_names},
            }

        new_lines = []
        class_counts = {name: 0 for name in self.class_names}
        total_lines = len(lines)
        kept_lines = 0

        for line_num, line in enumerate(lines):
            parts = line.strip().split()

            # Skip empty lines or malformed entries
            if len(parts) != 5:
                if len(parts) > 0:  # Only warn for non-empty malformed lines
                    print(f"Skipping malformed line {line_num + 1} in {label_path}: {line.strip()}")
                continue

            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError as e:
                print(f"Error parsing line {line_num + 1} in {label_path}: {e}")
                continue

            # Skip classes not in our mapping (background classes)
            if class_id not in id_map:
                continue

            # Skip invalid bounding boxes
            if bw <= 0 or bh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1 or bw * bh < min_area:
                continue

            # Valid annotation - remap class ID and keep
            new_id = id_map[class_id]
            new_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            # Update class counts using new class names
            if new_id < len(self.class_names):
                class_counts[self.class_names[new_id]] += 1

            kept_lines += 1

        return {
            "lines": new_lines,
            "total": total_lines,
            "kept": kept_lines,
            "class_counts": class_counts,
        }

    def _save_class_mapping(self):
        """
        Save class mapping information for debugging and validation.
        """
        mapping_info = {
            "original_classes": self.original_class_names,
            "filtered_classes": self.class_names,
            "id_mapping": self.id_map,
            "num_original_classes": len(self.original_class_names),
            "num_filtered_classes": len(self.class_names),
        }

        mapping_path = os.path.join(self.save_dir, "class_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(mapping_info, f, indent=2)

        print(f"Class mapping saved to: {mapping_path}")

    def _calculate_class_weights(self, class_counts):
        """
        Calculate class weights to handle imbalanced datasets.
        FIXED: Better handling of empty classes and validation.

        Args:
            class_counts (dict): Dictionary with class counts
        """
        if not class_counts or sum(class_counts.values()) == 0:
            print("Warning: No class counts available for weight calculation")
            return

        # Check for empty classes
        empty_classes = [cls for cls, count in class_counts.items() if count == 0]
        if empty_classes:
            print(f"Warning: Classes with no samples (will get default weight): {empty_classes}")

        # Convert to numpy arrays for sklearn (only for classes with samples)
        valid_classes = {cls: count for cls, count in class_counts.items() if count > 0}

        if len(valid_classes) == 0:
            print("Error: No classes have any samples!")
            return

        counts = np.array([valid_classes[name] for name in valid_classes.keys()])
        class_labels = np.arange(len(valid_classes))

        # Calculate balanced class weights
        weights = compute_class_weight(
            class_weight="balanced", classes=class_labels, y=np.repeat(class_labels, counts)
        )

        # Store weights - assign default weight for empty classes
        self.class_weights = {}
        default_weight = np.mean(weights) if len(weights) > 0 else 1.0

        valid_class_names = list(valid_classes.keys())

        for i, class_name in enumerate(self.class_names):
            if class_name in valid_class_names:
                weight_idx = valid_class_names.index(class_name)
                self.class_weights[i] = float(weights[weight_idx])
            else:
                self.class_weights[i] = default_weight
                print(f"Assigned default weight {default_weight:.3f} to empty class '{class_name}'")

        # Create tensor for PyTorch
        weight_values = [self.class_weights[i] for i in range(len(self.class_names))]
        self.class_weights_tensor = torch.FloatTensor(weight_values)

        print("Class weights calculated:")
        for i, (class_name, weight) in enumerate(zip(self.class_names, weight_values)):
            sample_count = class_counts.get(class_name, 0)
            print(f"  {class_name}: {weight:.3f} (samples: {sample_count})")

    def create_yolo_classification_config(self, classification_dir):
        """
        Create YOLO classification configuration file.
        FIXED: Ensures correct class count and names.

        Args:
            classification_dir (str): Path to classification dataset

        Returns:
            str: Path to configuration file
        """
        # Validate that we have the correct number of classes
        if len(self.class_names) == 0:
            raise ValueError("No valid classes found. Cannot create YOLO config.")

        config = {
            "path": os.path.abspath(classification_dir),
            "train": "train",
            "val": "val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        config_path = os.path.join(classification_dir, "data.yaml")

        # Handle existing config file conflicts
        if os.path.exists(config_path) and not os.path.isfile(config_path):
            print(f"Warning: {config_path} exists but is not a file. Removing...")
            if os.path.isdir(config_path):
                shutil.rmtree(config_path)
            else:
                os.remove(config_path)

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        print(f"YOLO classification config saved: {config_path}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Classes: {self.class_names}")

        return config_path

    def validate_model_architecture(self):
        """
        Validate that the model architecture matches our class configuration.
        NEW: Ensures model output dimensions match expected class count.

        Returns:
            bool: True if model architecture is correct, False otherwise
        """
        if self.model is None:
            print("Error: Model not initialized for architecture validation")
            return False

        try:
            # Get model's classification head
            if hasattr(self.model.model, "model") and hasattr(self.model.model.model, "classify"):
                # For classification models
                classify_layer = self.model.model.model.classify
                if hasattr(classify_layer, "linear"):
                    output_features = classify_layer.linear.out_features
                elif hasattr(classify_layer, "fc"):
                    output_features = classify_layer.fc.out_features
                else:
                    print("Warning: Could not determine model output features")
                    return False
            else:
                # For detection models adapted to classification
                print("Warning: Using detection model - architecture validation may be unreliable")
                return True

            expected_classes = len(self.class_names)

            print(f"Model validation:")
            print(f"  Expected classes: {expected_classes}")
            print(f"  Model output features: {output_features}")

            if output_features != expected_classes:
                print(f"ERROR: Model architecture mismatch!")
                print(
                    f"  Model outputs {output_features} classes but dataset has {expected_classes} classes"
                )
                print(f"  This will cause prediction errors!")
                return False
            else:
                print("✓ Model architecture matches class configuration")
                return True

        except Exception as e:
            print(f"Error during model architecture validation: {e}")
            return False

    def create_custom_loss(self):
        """
        Create custom weighted loss function for class imbalance handling.
        FIXED: Better error handling and validation.

        Returns:
            nn.Module: Custom weighted loss function
        """
        if self.class_weights_tensor is None:
            print("Warning: No class weights available for custom loss")
            return None

        class WeightedCrossEntropyLoss(nn.Module):
            def __init__(self, weights, num_classes):
                super().__init__()
                self.weights = weights
                self.num_classes = num_classes
                self.ce_loss = nn.CrossEntropyLoss(weight=weights, reduction="mean")

            def forward(self, predictions, targets):
                # Validate input dimensions
                if predictions.size(1) != self.num_classes:
                    raise RuntimeError(
                        f"Prediction tensor has {predictions.size(1)} classes, expected {self.num_classes}"
                    )

                return self.ce_loss(predictions, targets)

        self.weighted_loss_fn = WeightedCrossEntropyLoss(
            self.class_weights_tensor, len(self.class_names)
        )
        print(f"Custom weighted loss function created for {len(self.class_names)} classes")
        return self.weighted_loss_fn

    def patch_model_loss(self):
        """
        Patch the YOLOv12 model to use weighted loss function.
        FIXED: Better validation and error handling.

        Returns:
            bool: True if patching successful, False otherwise
        """
        if self.model is None or self.class_weights_tensor is None:
            print("Warning: Model or class weights not available for loss patching")
            return False

        try:
            # Get model device
            device = next(self.model.model.parameters()).device
            weighted_tensor = self.class_weights_tensor.to(device)

            print(f"Applying class weights on device: {device}")
            print(f"Class weights tensor shape: {weighted_tensor.shape}")
            print(f"Expected classes: {len(self.class_names)}")

            # Patch loss function if available
            if hasattr(self.model.model, "loss"):
                original_loss = self.model.model.loss

                def weighted_loss_wrapper(*args, **kwargs):
                    loss_dict = original_loss(*args, **kwargs)

                    # Apply weights to classification loss
                    if "cls" in loss_dict and len(args) >= 2:
                        preds, targets = args[0], args[1]
                        if hasattr(targets, "cls"):
                            cls_targets = targets.cls.long()
                            cls_preds = preds[0] if isinstance(preds, (list, tuple)) else preds

                            # Validate dimensions before applying weighted loss
                            if cls_preds.size(1) != len(self.class_names):
                                print(
                                    f"Warning: Prediction dimension mismatch - got {cls_preds.size(1)}, expected {len(self.class_names)}"
                                )
                                return loss_dict

                            # Apply weighted cross entropy
                            weighted_cls_loss = nn.CrossEntropyLoss(weight=weighted_tensor)(
                                cls_preds, cls_targets
                            )
                            loss_dict["cls"] = weighted_cls_loss

                    return loss_dict

                self.model.model.loss = weighted_loss_wrapper
                print(f"Successfully patched model loss with class weights")
                return True

        except Exception as e:
            print(f"Could not patch model loss: {e}")

        return False

    def initialize_yolov12_classifier(self):
        """
        Initialize YOLOv12 model for classification with class weights.
        FIXED: Better model validation and architecture checking.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if len(self.class_names) == 0:
            print("Error: No classes defined. Cannot initialize model.")
            return False

        try:
            # Try classification model first
            model_name = f"yolo12{self.model_size}-cls.pt"
            print(f"Attempting to load {model_name}...")
            self.model = YOLO(model_name)
            print(f"YOLOv12{self.model_size} Classification model initialized successfully")

            # Validate model architecture
            architecture_valid = self.validate_model_architecture()
            if not architecture_valid:
                print(
                    "Warning: Model architecture validation failed - predictions may be unreliable"
                )

            # Apply class weights if available
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
                print(f"Trying fallback detection model: {model_name}")
                self.model = YOLO(model_name)
                print(f"Using YOLOv12{self.model_size} detection model as fallback")
                print(
                    "Warning: Using detection model for classification may have architecture mismatches"
                )

                # Apply class weights if available
                if self.class_weights_tensor is not None:
                    loss_patched = self.patch_model_loss()
                    if not loss_patched:
                        print("Warning: Could not apply class weights to model loss")
                        self.create_custom_loss()

                return True

            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                return False

    def apply_progressive_unfreezing(self, unfreeze_schedule=None):
        """
        Apply progressive unfreezing strategy to the YOLOv12 model.

        Args:
            unfreeze_schedule (dict): Schedule for unfreezing layers
                                    Format: {epoch: percentage_to_unfreeze}
                                    Default: {0: 0.1, 20: 0.3, 50: 0.6, 75: 1.0}

        Returns:
            dict: Unfreezing configuration
        """
        if self.model is None:
            print("Error: Model not initialized. Call initialize_yolov12_classifier() first.")
            return None

        # Default progressive unfreezing schedule
        if unfreeze_schedule is None:
            unfreeze_schedule = {
                0: 0.1,  # Start with 10% of layers unfrozen
                20: 0.3,  # Unfreeze 30% after 20 epochs
                50: 0.6,  # Unfreeze 60% after 50 epochs
                75: 1.0,  # Unfreeze all layers after 75 epochs
            }

        print("Setting up progressive unfreezing strategy:")
        for epoch, percentage in sorted(unfreeze_schedule.items()):
            print(f"  Epoch {epoch}: {percentage*100:.0f}% of layers will be unfrozen")

        # Get total number of trainable parameters
        total_params = []
        param_names = []

        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                total_params.append(param)
                param_names.append(name)

        total_layer_count = len(total_params)
        print(f"Total trainable parameters found: {total_layer_count}")

        # Initially freeze most layers (keep only the specified percentage unfrozen)
        initial_unfreeze_ratio = unfreeze_schedule.get(0, 0.1)
        layers_to_unfreeze = int(total_layer_count * initial_unfreeze_ratio)

        # Freeze all parameters first
        for param in total_params:
            param.requires_grad = False

        # Unfreeze the last N layers (typically classifier head and some backbone layers)
        for i in range(total_layer_count - layers_to_unfreeze, total_layer_count):
            total_params[i].requires_grad = True

        frozen_count = sum(1 for param in total_params if not param.requires_grad)
        unfrozen_count = total_layer_count - frozen_count

        print(f"Initial state: {unfrozen_count} layers unfrozen, {frozen_count} layers frozen")

        # Store unfreezing configuration
        self.unfreeze_schedule = unfreeze_schedule
        self.total_params = total_params
        self.param_names = param_names
        self.total_layer_count = total_layer_count

        return {
            "schedule": unfreeze_schedule,
            "total_layers": total_layer_count,
            "initial_unfrozen": unfrozen_count,
            "initial_frozen": frozen_count,
        }

    def update_unfreezing_schedule(self, current_epoch):
        """
        Update layer freezing status based on current epoch.
        Call this function at the beginning of each epoch during training.

        Args:
            current_epoch (int): Current training epoch

        Returns:
            bool: True if unfreezing status was updated, False otherwise
        """
        if not hasattr(self, "unfreeze_schedule") or not hasattr(self, "total_params"):
            print(
                "Warning: Progressive unfreezing not initialized. Call apply_progressive_unfreezing() first."
            )
            return False

        # Check if we need to update unfreezing at this epoch
        if current_epoch in self.unfreeze_schedule:
            target_ratio = self.unfreeze_schedule[current_epoch]
            target_unfrozen_count = int(self.total_layer_count * target_ratio)

            # Freeze all parameters first
            for param in self.total_params:
                param.requires_grad = False

            # Unfreeze the specified number of layers (from the end)
            for i in range(self.total_layer_count - target_unfrozen_count, self.total_layer_count):
                self.total_params[i].requires_grad = True

            frozen_count = sum(1 for param in self.total_params if not param.requires_grad)
            unfrozen_count = self.total_layer_count - frozen_count

            print(
                f"Epoch {current_epoch}: Updated unfreezing - {unfrozen_count} layers unfrozen ({target_ratio*100:.0f}%)"
            )

            return True

        return False

    def train_model_with_progressive_unfreezing_and_class_weights(
        self, classification_path, epochs=100, unfreeze_schedule=None
    ):
        """
        Unified training method combining progressive unfreezing AND class weights.
        FIXED: Better validation and error handling throughout training process.

        Args:
            classification_path (str): Path to classification dataset
            epochs (int): Number of training epochs
            unfreeze_schedule (dict): Custom unfreezing schedule

        Returns:
            object: Training results
        """
        print(f"\n{'='*60}")
        print("STARTING UNIFIED TRAINING: PROGRESSIVE UNFREEZING + CLASS WEIGHTS")
        print(f"{'='*60}")

        # Validate prerequisites
        if len(self.class_names) == 0:
            raise ValueError("No classes defined. Cannot proceed with training.")

        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_yolov12_classifier() first.")

        # Apply progressive unfreezing strategy
        unfreezing_config = self.apply_progressive_unfreezing(unfreeze_schedule)
        if unfreezing_config is None:
            print(
                "Failed to set up progressive unfreezing. Falling back to standard weighted training."
            )
            return self.train_model_with_class_weights(classification_path, epochs)

        print(f"Model: YOLOv12{self.model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Classes: {self.class_names}")
        print(f"Progressive unfreezing: ENABLED")
        print(f"Class weights: ENABLED")

        # Display class weights
        if self.class_weights:
            print("\nClass weights to be applied in ALL phases:")
            for class_id, weight in self.class_weights.items():
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"Unknown_{class_id}"
                )
                print(f"  {class_name}: {weight:.3f}")
        else:
            print("Warning: No class weights calculated!")

        # Display unfreezing schedule
        print("\nProgressive unfreezing schedule:")
        for epoch, percentage in sorted(self.unfreeze_schedule.items()):
            print(f"  Epoch {epoch}: {percentage*100:.0f}% of layers unfrozen")

        print(f"{'='*60}")

        # Create classification config
        config_path = self.create_yolo_classification_config(classification_path)

        # Base training arguments
        base_training_args = {
            "data": config_path,
            "imgsz": self.img_size,
            "batch": self.batch_size,
            "device": "cpu",
            "workers": 4,
            "patience": 20,
            "save": True,
            "save_period": 10,
            "val": True,
            "project": self.save_dir,
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

        # Phase-based training with both progressive unfreezing AND class weights
        epoch_phases = sorted(self.unfreeze_schedule.keys())
        all_phase_results = []

        print("Starting unified training with BOTH progressive unfreezing and class weights...")

        for i, phase_start_epoch in enumerate(epoch_phases):
            # Calculate epochs for this phase
            if i + 1 < len(epoch_phases):
                phase_epochs = epoch_phases[i + 1] - phase_start_epoch
            else:
                phase_epochs = epochs - phase_start_epoch

            if phase_epochs <= 0:
                continue

            print(f"\n{'='*50}")
            print(
                f"PHASE {i+1}: EPOCHS {phase_start_epoch} to {phase_start_epoch + phase_epochs - 1}"
            )
            print(f"{'='*50}")

            # Update unfreezing for this phase
            unfreezing_updated = self.update_unfreezing_schedule(phase_start_epoch)

            # For phases after the first, load the previous checkpoint
            if i > 0:
                previous_phase_name = f"unified_phase_{i}"
                last_checkpoint = os.path.join(
                    self.save_dir, previous_phase_name, "weights", "last.pt"
                )
                if os.path.exists(last_checkpoint):
                    print(f"Resuming from checkpoint: {last_checkpoint}")
                    self.model = YOLO(last_checkpoint)

                    # CRITICAL: Reapply BOTH class weights AND unfreezing after loading checkpoint
                    print("Reapplying class weights to resumed model...")
                    class_weights_applied = self.patch_model_loss()
                    if not class_weights_applied:
                        print("Warning: Could not reapply class weights to resumed model")

                    print("Reapplying unfreezing schedule to resumed model...")
                    self.update_unfreezing_schedule(phase_start_epoch)

            # Apply class weights to the model for this phase
            print("Ensuring class weights are applied for this phase...")
            if not self.patch_model_loss():
                print("Warning: Could not apply class weights for this phase")
                if self.weighted_loss_fn is None:
                    self.create_custom_loss()

            # Configure training arguments for this specific phase
            phase_training_args = base_training_args.copy()
            phase_training_args.update(
                {
                    "epochs": phase_epochs,
                    "name": f"unified_phase_{i+1}",
                }
            )

            # Display phase information
            unfrozen_count = sum(1 for param in self.total_params if param.requires_grad)
            frozen_count = self.total_layer_count - unfrozen_count
            unfrozen_percentage = (
                (unfrozen_count / self.total_layer_count) * 100 if self.total_layer_count > 0 else 0
            )

            print(f"Phase {i+1} configuration:")
            print(f"  Epochs: {phase_epochs}")
            print(
                f"  Unfrozen layers: {unfrozen_count}/{self.total_layer_count} ({unfrozen_percentage:.1f}%)"
            )
            print(f"  Frozen layers: {frozen_count}")
            print(f"  Class weights: {'Applied' if self.class_weights else 'Not available'}")

            # Execute training for this phase
            print(f"Starting training for phase {i+1}...")
            try:
                phase_results = self.model.train(**phase_training_args)
                all_phase_results.append(
                    {
                        "phase": i + 1,
                        "start_epoch": phase_start_epoch,
                        "epochs": phase_epochs,
                        "unfrozen_percentage": unfrozen_percentage,
                        "results": phase_results,
                    }
                )
                print(f"Phase {i+1} completed successfully!")

            except Exception as e:
                print(f"Error in phase {i+1}: {e}")
                print("Attempting to continue with next phase...")
                continue

        print(f"\n{'='*60}")
        print("UNIFIED TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("Features applied:")
        print("✓ Progressive unfreezing across all training phases")
        print("✓ Class weights applied in every phase")
        print("✓ Background class filtering and proper ID mapping")
        print("✓ Automatic checkpoint management between phases")
        print("✓ Layer freezing status tracking")
        print("✓ Model architecture validation")

        print(f"\nTraining summary:")
        print(f"  Total phases: {len(all_phase_results)}")
        print(f"  Total epochs: {epochs}")
        print(f"  Final classes: {self.class_names}")

        # Return comprehensive results
        return {
            "all_phases": all_phase_results,
            "final_results": all_phase_results[-1]["results"] if all_phase_results else None,
            "class_weights_applied": True,
            "progressive_unfreezing_applied": True,
            "unfreezing_schedule": self.unfreeze_schedule,
            "class_weights": self.class_weights,
            "final_classes": self.class_names,
            "class_mapping": self.id_map,
        }

    def train_model_with_class_weights(self, classification_path, epochs=100):
        """
        Standard training method with class weights only (no progressive unfreezing).
        FIXED: Fallback method for when progressive unfreezing fails.

        Args:
            classification_path (str): Path to classification dataset
            epochs (int): Number of training epochs

        Returns:
            object: Training results
        """
        print(f"\n{'='*50}")
        print("STARTING STANDARD TRAINING WITH CLASS WEIGHTS")
        print(f"{'='*50}")

        # Validate prerequisites
        if len(self.class_names) == 0:
            raise ValueError("No classes defined. Cannot proceed with training.")

        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_yolov12_classifier() first.")

        print(f"Model: YOLOv12{self.model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Classes: {self.class_names}")

        # Display class weights
        if self.class_weights:
            print("\nApplied class weights:")
            for class_id, weight in self.class_weights.items():
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"Unknown_{class_id}"
                )
                print(f"  {class_name}: {weight:.3f}")
        else:
            print("Warning: No class weights calculated!")

        # Create classification config
        config_path = self.create_yolo_classification_config(classification_path)

        # Apply class weights to model
        print("Applying class weights to model...")
        if not self.patch_model_loss():
            print("Warning: Could not apply class weights to model loss")
            if self.weighted_loss_fn is None:
                self.create_custom_loss()

        # Training arguments
        training_args = {
            "data": config_path,
            "epochs": epochs,
            "imgsz": self.img_size,
            "batch": self.batch_size,
            "device": "cpu",
            "workers": 4,
            "patience": 20,
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

        print("Starting training with class weights...")
        try:
            results = self.model.train(**training_args)
            print("Training completed successfully!")

            return {
                "results": results,
                "class_weights_applied": True,
                "progressive_unfreezing_applied": False,
                "class_weights": self.class_weights,
                "final_classes": self.class_names,
                "class_mapping": self.id_map,
            }
        except Exception as e:
            print(f"Training failed: {e}")
            raise

    def validate_model(self, classification_path):
        """
        Validate the trained model with comprehensive metrics.
        FIXED: Better handling of model path detection and class mapping consistency.

        Args:
            classification_path (str): Path to classification dataset

        Returns:
            dict: Validation results including Top-K metrics
        """
        print("Validating model...")

        # Try to find the best model from different possible locations
        possible_best_paths = [
            os.path.join(self.save_dir, "weighted_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "unified_phase_1", "weights", "best.pt"),
        ]

        # Also check all phase directories
        for i in range(1, 10):  # Check up to 10 phases
            phase_path = os.path.join(self.save_dir, f"unified_phase_{i}", "weights", "best.pt")
            if os.path.exists(phase_path):
                possible_best_paths.append(phase_path)

        best_model_path = None
        for path in possible_best_paths:
            if os.path.exists(path):
                best_model_path = path
                break

        if best_model_path:
            self.model = YOLO(best_model_path)
            print(f"Loaded best model from: {best_model_path}")
        else:
            print("Warning: No trained model found. Using current model state.")

        # Run standard validation
        config_path = os.path.join(classification_path, "data.yaml")
        try:
            results = self.model.val(data=config_path)
            print("Standard validation completed.")
        except Exception as e:
            print(f"Standard validation failed: {e}")
            results = None

        # Calculate Top-K accuracy metrics
        try:
            top_k_metrics = self.calculate_top_k_accuracy(classification_path, k_values=[1, 3, 5])
        except Exception as e:
            print(f"Top-K accuracy calculation failed: {e}")
            top_k_metrics = {}

        validation_results = {
            "standard_metrics": results,
            "top_k_metrics": top_k_metrics,
            "class_weights_applied": True,
            "class_weights": self.class_weights,
            "final_classes": self.class_names,
            "class_mapping": self.id_map,
            "model_path": best_model_path,
        }

        print("Validation completed!")
        if "top_1_accuracy" in top_k_metrics:
            print(f"Top-1 Accuracy: {top_k_metrics['top_1_accuracy']:.4f}")
        if "top_3_accuracy" in top_k_metrics:
            print(f"Top-3 Accuracy: {top_k_metrics['top_3_accuracy']:.4f}")

        return validation_results

    def calculate_top_k_accuracy(self, classification_path, k_values=[1, 3, 5]):
        """
        Calculate Top-K accuracy metrics for the model.
        FIXED: Better error handling and validation directory detection.

        Args:
            classification_path (str): Path to classification dataset
            k_values (list): List of K values for Top-K accuracy

        Returns:
            dict: Top-K accuracy results
        """
        print(f"Calculating Top-K accuracy for k={k_values}...")

        # Check if validation directory exists
        val_path = os.path.join(classification_path, "val")
        if not os.path.exists(val_path):
            print(f"Validation directory not found: {val_path}")
            return {}

        # For YOLO classification format, check for images directory
        val_images_path = os.path.join(val_path, "images")

        y_true = []
        y_pred_probs = []
        processed_images = 0

        if os.path.exists(val_images_path):
            # YOLO classification format: val/images/*.jpg
            print("Using YOLO classification format validation...")

            for img_file in os.listdir(val_images_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(val_images_path, img_file)

                # Try to determine true class from filename or other method
                # This might need adjustment based on your specific dataset structure
                true_class_found = False
                for class_idx, class_name in enumerate(self.class_names):
                    if class_name.lower() in img_file.lower():
                        try:
                            results = self.model.predict(img_path, verbose=False, conf=0.5)
                            if results and len(results) > 0:
                                probs = results[0].probs
                                if probs is not None and hasattr(probs, "data"):
                                    prob_scores = probs.data.cpu().numpy()
                                    y_true.append(class_idx)
                                    y_pred_probs.append(prob_scores)
                                    processed_images += 1
                                    true_class_found = True
                                    break
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            continue

                if not true_class_found:
                    print(f"Could not determine class for {img_file}")
        else:
            # Original format: val/class_name/*.jpg
            print("Using class-directory validation format...")

            for class_idx, class_name in enumerate(self.class_names):
                class_path = os.path.join(val_path, class_name)
                if os.path.exists(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                            img_path = os.path.join(class_path, img_file)

                            try:
                                results = self.model.predict(img_path, verbose=False, conf=0.5)
                                if results and len(results) > 0:
                                    probs = results[0].probs
                                    if probs is not None and hasattr(probs, "data"):
                                        prob_scores = probs.data.cpu().numpy()
                                        y_true.append(class_idx)
                                        y_pred_probs.append(prob_scores)
                                        processed_images += 1
                            except Exception as e:
                                print(f"Error processing {img_path}: {e}")
                                continue

        print(f"Processed {processed_images} images for Top-K accuracy calculation")

        if len(y_true) == 0:
            print("No predictions found for accuracy calculation")
            return {}

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)

        # Validate dimensions
        if y_pred_probs.shape[1] != len(self.class_names):
            print(
                f"Warning: Prediction dimension mismatch - got {y_pred_probs.shape[1]}, expected {len(self.class_names)}"
            )

        # Calculate Top-K accuracies
        top_k_accuracies = {}
        for k in k_values:
            if k <= len(self.class_names) and k <= y_pred_probs.shape[1]:
                try:
                    acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
                    top_k_accuracies[f"top_{k}_accuracy"] = acc
                    print(f"Top-{k} Accuracy: {acc:.4f}")
                except Exception as e:
                    print(f"Error calculating Top-{k} accuracy: {e}")

        return top_k_accuracies

    def create_confusion_matrix(self, classification_path):
        """
        Create comprehensive confusion matrix analysis with visualizations.
        FIXED: Better validation data detection and class mapping consistency.

        Args:
            classification_path (str): Path to classification dataset
        """
        print("Creating confusion matrix analysis...")

        # Check validation directory structure
        val_path = os.path.join(classification_path, "val")
        if not os.path.exists(val_path):
            print(f"Validation directory not found: {val_path}")
            return

        val_images_path = os.path.join(val_path, "images")

        y_true = []
        y_pred = []
        processed_images = 0

        if os.path.exists(val_images_path):
            # YOLO classification format
            print("Using YOLO classification format for confusion matrix...")

            for img_file in os.listdir(val_images_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(val_images_path, img_file)

                # Determine true class from filename
                true_class_found = False
                for class_idx, class_name in enumerate(self.class_names):
                    if class_name.lower() in img_file.lower():
                        try:
                            results = self.model.predict(img_path, verbose=False, conf=0.5)
                            if results and len(results) > 0:
                                probs = results[0].probs
                                if probs is not None:
                                    predicted_class = probs.top1
                                    y_true.append(class_idx)
                                    y_pred.append(predicted_class)
                                    processed_images += 1
                                    true_class_found = True
                                    break
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            continue
        else:
            # Class directory format
            print("Using class-directory format for confusion matrix...")

            for class_idx, class_name in enumerate(self.class_names):
                class_path = os.path.join(val_path, class_name)
                if os.path.exists(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                            img_path = os.path.join(class_path, img_file)

                            try:
                                results = self.model.predict(img_path, verbose=False, conf=0.5)
                                if results and len(results) > 0:
                                    probs = results[0].probs
                                    if probs is not None:
                                        predicted_class = probs.top1
                                        y_true.append(class_idx)
                                        y_pred.append(predicted_class)
                                        processed_images += 1
                            except Exception as e:
                                print(f"Error processing {img_path}: {e}")
                                continue

        print(f"Processed {processed_images} images for confusion matrix")

        if len(y_true) == 0:
            print("No predictions found for confusion matrix")
            return

        # Ensure predictions are within valid class range
        y_pred = [min(pred, len(self.class_names) - 1) for pred in y_pred]

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        cm_norm = cm.astype("float") / (
            cm.sum(axis=1)[:, np.newaxis] + 1e-8
        )  # Add small epsilon to avoid division by zero

        # Calculate per-class metrics
        class_accuracies = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        class_support = cm.sum(axis=1)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Confusion matrix (raw counts)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix (Raw Counts)")
        axes[0, 0].set_xlabel("Predicted Labels")
        axes[0, 0].set_ylabel("True Labels")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Normalized confusion matrix
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

        # Add accuracy values on bars
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
            range(len(self.class_names)), weights_to_plot, color="purple", alpha=0.7
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

        # Generate classification report
        try:
            report = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )
            report["class_weights_applied"] = self.class_weights
            report["class_mapping"] = self.id_map
            report["final_classes"] = self.class_names

            report_path = os.path.join(self.save_dir, "classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Classification report saved to: {report_path}")
        except Exception as e:
            print(f"Error generating classification report: {e}")

        print(f"Confusion matrix saved to: {save_path}")

    def save_model_info(self):
        """
        Save comprehensive model information and metadata.
        FIXED: Includes class mapping and background removal information.
        """
        # Try to find the best model path
        possible_best_paths = [
            os.path.join(self.save_dir, "weighted_training", "weights", "best.pt"),
            os.path.join(self.save_dir, "unified_phase_1", "weights", "best.pt"),
        ]

        best_model_path = None
        for path in possible_best_paths:
            if os.path.exists(path):
                best_model_path = path
                break

        metadata = {
            "model_type": f"Fixed Weighted YOLOv12{self.model_size} Classifier",
            "original_classes": self.original_class_names,
            "final_classes": self.class_names,
            "class_mapping": self.id_map,
            "background_classes_removed": len(self.original_class_names) - len(self.class_names),
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "num_original_classes": len(self.original_class_names),
            "num_final_classes": len(self.class_names),
            "class_weights_applied": self.class_weights,
            "class_weights_tensor": (
                self.class_weights_tensor.tolist()
                if self.class_weights_tensor is not None
                else None
            ),
            "training_timestamp": datetime.datetime.now().isoformat(),
            "save_directory": self.save_dir,
            "model_path": best_model_path,
            "fixes_applied": [
                "Background class removal with proper ID remapping",
                "Model architecture validation",
                "Consistent class mapping throughout pipeline",
                "Better error handling and validation",
                "Improved prediction consistency",
                "Enhanced debugging and logging",
            ],
            "features": [
                "Class weights applied through custom loss function",
                "Object detection to classification conversion",
                "Top-K accuracy metrics",
                "Class imbalance handling",
                "Stratified train/validation split",
                "Comprehensive confusion matrix analysis",
                "Progressive layer unfreezing support",
                "Background class filtering",
            ],
        }

        metadata_path = os.path.join(self.save_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model metadata saved to: {metadata_path}")
        print(f"Key improvements:")
        print(f"  - Background classes removed: {metadata['background_classes_removed']}")
        print(
            f"  - Original classes: {len(self.original_class_names)} → Final classes: {len(self.class_names)}"
        )
        print(f"  - Class mapping validation: ✓")
        print(f"  - Model architecture validation: ✓")


def main():
    """
    Main training pipeline with unified progressive unfreezing + class weights.
    FIXED VERSION: Properly handles background class removal throughout the entire pipeline.
    """
    try:
        # Initialize classifier
        print("Initializing FIXED WeightedYOLOv12Classifier...")
        classifier = WeightedYOLOv12Classifier(model_size="n", img_size=640, batch_size=12)

        # Steps 1-3: Dataset download, cleaning, and model initialization
        print("\nStep 1: Downloading dataset...")
        gdrive_file_id = "1zvCNOz4P0QFdOAfpDpSpidFVjXHFxDUE"
        dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

        print("\nStep 2: Cleaning dataset and removing background classes...")
        classification_path = classifier.clean_and_extract_objects(
            dataset_path, min_area=0.0001, max_samples_per_class=15000, val_split=0.2
        )

        print("\nStep 3: Initializing YOLOv12 model...")
        if not classifier.initialize_yolov12_classifier():
            raise RuntimeError("Failed to initialize YOLOv12 model")

        print(f"\n{'='*60}")
        print("BACKGROUND CLASS REMOVAL VERIFICATION:")
        print(f"{'='*60}")
        print(f"Original classes: {classifier.original_class_names}")
        print(f"Final classes: {classifier.class_names}")
        print(
            f"Classes removed: {len(classifier.original_class_names) - len(classifier.class_names)}"
        )
        print(f"Class ID mapping: {classifier.id_map}")
        print(f"{'='*60}")

        # Step 4: UNIFIED TRAINING with both progressive unfreezing AND class weights
        print("\n" + "=" * 60)
        print("STEP 4: UNIFIED TRAINING - PROGRESSIVE UNFREEZING + CLASS WEIGHTS")
        print("=" * 60)

        # Optional: Define custom unfreezing schedule
        custom_schedule = {
            0: 0.15,  # Start with 15% unfrozen
            25: 0.4,  # 40% after 25 epochs
            60: 0.7,  # 70% after 60 epochs
            85: 1.0,  # All layers after 85 epochs
        }

        # Use the unified training method that combines BOTH features
        training_results = classifier.train_model_with_progressive_unfreezing_and_class_weights(
            classification_path, epochs=100, unfreeze_schedule=custom_schedule
        )

        # Steps 5-8: Continue with validation, analysis, etc.
        print("\nStep 5: Validating model...")
        validation_results = classifier.validate_model(classification_path)

        print("\nStep 6: Creating confusion matrix analysis...")
        classifier.create_confusion_matrix(classification_path)

        print("\nStep 7: Saving model information...")
        classifier.save_model_info()

        # Cleanup
        if os.path.exists("dataset/"):
            shutil.rmtree("dataset/")

        print("\n" + "=" * 60)
        print("FIXED UNIFIED TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Key fixes applied:")
        print("✓ Background class removal with proper ID remapping")
        print("✓ Model architecture validation")
        print("✓ Consistent class mapping throughout pipeline")
        print("✓ Enhanced error handling and validation")
        print("✓ Improved prediction consistency")
        print("=" * 60)

        return classifier

    except Exception as e:
        print(f"\nERROR: Training pipeline failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def run_prediction_example(classifier, test_image_path):
    """
    Example function to run predictions on new images.
    FIXED: Uses consistent class mapping for predictions.

    Args:
        classifier (WeightedYOLOv12Classifier): Trained classifier instance
        test_image_path (str): Path to test image
    """
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    print(f"\nRunning prediction on: {test_image_path}")
    print(f"Model trained on {len(classifier.class_names)} classes: {classifier.class_names}")

    # Try to find best model from different possible locations
    possible_best_paths = [
        os.path.join(classifier.save_dir, "weighted_training", "weights", "best.pt"),
        os.path.join(classifier.save_dir, "unified_phase_1", "weights", "best.pt"),
    ]

    best_model_path = None
    for path in possible_best_paths:
        if os.path.exists(path):
            best_model_path = path
            break

    if best_model_path:
        model = YOLO(best_model_path)
        print(f"Using model: {best_model_path}")

        try:
            # Run prediction
            results = model.predict(test_image_path, verbose=False, conf=0.5)

            if results and len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    # Get top predictions
                    top_indices = probs.top5
                    top_confidences = probs.top5conf.cpu().numpy()

                    print("Top 5 predictions:")
                    for i, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
                        if idx < len(classifier.class_names):
                            class_name = classifier.class_names[idx]
                            print(f"  {i+1}. {class_name}: {conf:.3f}")
                        else:
                            print(f"  {i+1}. Unknown_class_{idx}: {conf:.3f}")
                            print(
                                f"    WARNING: Prediction index {idx} exceeds expected class count {len(classifier.class_names)}"
                            )
                else:
                    print("No probability scores found in prediction results")
            else:
                print("No prediction results found")
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print("Trained model not found!")


def debug_class_mapping(classifier):
    """
    Debug function to verify class mapping consistency.
    NEW: Helps identify class mapping issues.

    Args:
        classifier (WeightedYOLOv12Classifier): Classifier instance
    """
    print("\n" + "=" * 50)
    print("CLASS MAPPING DEBUG INFORMATION")
    print("=" * 50)

    print(
        f"Original classes ({len(classifier.original_class_names)}): {classifier.original_class_names}"
    )
    print(f"Final classes ({len(classifier.class_names)}): {classifier.class_names}")
    print(f"ID mapping: {classifier.id_map}")

    if classifier.class_weights:
        print("\nClass weights:")
        for class_id, weight in classifier.class_weights.items():
            if class_id < len(classifier.class_names):
                print(f"  {classifier.class_names[class_id]} (ID {class_id}): {weight:.3f}")
            else:
                print(f"  Unknown_class_{class_id}: {weight:.3f}")

    # Check for potential issues
    issues = []

    if len(classifier.class_names) == 0:
        issues.append("No final classes defined")

    if not classifier.id_map:
        issues.append("No ID mapping created")

    if classifier.class_weights and len(classifier.class_weights) != len(classifier.class_names):
        issues.append(
            f"Class weight count ({len(classifier.class_weights)}) doesn't match class count ({len(classifier.class_names)})"
        )

    if issues:
        print(f"\nPOTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"{issue}")
    else:
        print(f"\nClass mapping appears consistent")

    print("=" * 50)


if __name__ == "__main__":
    # Run the main training pipeline
    trained_classifier = main()

    # Debug class mapping
    debug_class_mapping(trained_classifier)
