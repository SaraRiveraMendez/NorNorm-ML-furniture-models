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
    """

    def __init__(self, model_size="n", img_size=640, batch_size=16):
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
        self.class_weights = None
        self.class_weights_tensor = None
        self.weighted_loss_fn = None

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
            self.class_names = data_config.get("names", [])

        # Clean class names and remove background classes
        original_names = [n.strip().lower() for n in self.class_names]
        self.class_names = [n for n in original_names if n not in ["background", "bg"]]

        if len(self.class_names) < len(original_names):
            print(f"Removed background class(es). {len(original_names)} → {len(self.class_names)}")

        print(f"Target classes: {self.class_names}")

        # Create mapping from old indices to new indices
        id_map = {
            i: self.class_names.index(n)
            for i, n in enumerate(original_names)
            if n in self.class_names
        }

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
            all_images, train_labels_dir, id_map, val_split
        )

        print(f"Split created: {len(train_imgs)} train, {len(val_imgs)} val")

        # Process training images
        train_stats = self._process_image_split(
            train_imgs,
            train_images_dir,
            train_labels_dir,
            new_train_images_dir,
            new_train_labels_dir,
            id_map,
            min_area,
        )

        # Process validation images
        val_stats = self._process_image_split(
            val_imgs,
            train_images_dir,
            train_labels_dir,
            new_val_images_dir,
            new_val_labels_dir,
            id_map,
            min_area,
        )

        # Combine statistics
        total_labels = train_stats["total"] + val_stats["total"]
        kept_labels = train_stats["kept"] + val_stats["kept"]

        for class_name in self.class_names:
            class_counts[class_name] = (
                train_stats["class_counts"][class_name] + val_stats["class_counts"][class_name]
            )

        # Calculate class weights based on object counts
        self._calculate_class_weights(class_counts)

        print(f"\nObject cleaning summary:")
        print(f"Original labels: {total_labels}")
        print(f"Kept labels: {kept_labels}")
        print(f"Train/Val split: {len(train_imgs)}/{len(val_imgs)}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} objects")

        return classification_dir

    def _create_stratified_split(self, all_images, train_labels_dir, id_map, val_split):
        """
        Create stratified train/validation split based on image classes.

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
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            if class_id in id_map:
                                img_class = id_map[class_id]
                                break

                    if img_class is not None:
                        valid_images.append(img_file)
                        image_classes.append(img_class)

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

        for img_file in images:
            img_path = os.path.join(src_img_dir, img_file)
            new_img_path = os.path.join(dst_img_dir, img_file)

            # Copy image
            shutil.copy2(img_path, new_img_path)

            # Process corresponding label file
            label_file = os.path.splitext(img_file)[0] + ".txt"
            old_label_path = os.path.join(src_labels_dir, label_file)
            new_label_path = os.path.join(dst_labels_dir, label_file)

            if os.path.exists(old_label_path):
                processed_labels = self._process_label_file(old_label_path, id_map, min_area)
                total_labels += processed_labels["total"]
                kept_labels += processed_labels["kept"]

                # Update class counts
                for class_name, count in processed_labels["class_counts"].items():
                    class_counts[class_name] += count

                # Write processed labels
                if processed_labels["lines"]:
                    with open(new_label_path, "w") as f:
                        f.writelines(processed_labels["lines"])

        return {"total": total_labels, "kept": kept_labels, "class_counts": class_counts}

    def _process_label_file(self, label_path, id_map, min_area):
        """
        Process individual label files by cleaning and remapping class IDs.

        Args:
            label_path (str): Path to label file
            id_map (dict): Class ID mapping
            min_area (float): Minimum bbox area threshold

        Returns:
            dict: Processing results
        """
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        class_counts = {name: 0 for name in self.class_names}
        total_lines = len(lines)
        kept_lines = 0

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            if class_id not in id_map:
                continue

            cx, cy, bw, bh = map(float, parts[1:])

            # Skip invalid bounding boxes
            if bw <= 0 or bh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1 or bw * bh < min_area:
                continue

            new_id = id_map[class_id]
            new_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            class_counts[self.class_names[new_id]] += 1
            kept_lines += 1

        return {
            "lines": new_lines,
            "total": total_lines,
            "kept": kept_lines,
            "class_counts": class_counts,
        }

    def _calculate_class_weights(self, class_counts):
        """
        Calculate class weights to handle imbalanced datasets.

        Args:
            class_counts (dict): Dictionary with class counts
        """
        if not class_counts or sum(class_counts.values()) == 0:
            print("Warning: No class counts available for weight calculation")
            return

        # Convert to numpy arrays for sklearn
        counts = np.array([class_counts[name] for name in self.class_names])
        class_labels = np.arange(len(self.class_names))

        # Calculate balanced class weights
        weights = compute_class_weight(
            class_weight="balanced", classes=class_labels, y=np.repeat(class_labels, counts)
        )

        # Store weights
        self.class_weights = {i: float(weight) for i, weight in enumerate(weights)}
        self.class_weights_tensor = torch.FloatTensor(weights)

        print("Class weights calculated:")
        for i, (class_name, weight) in enumerate(zip(self.class_names, weights)):
            print(f"  {class_name}: {weight:.3f}")

    def create_yolo_classification_config(self, classification_dir):
        """
        Create YOLO classification configuration file.

        Args:
            classification_dir (str): Path to classification dataset

        Returns:
            str: Path to configuration file
        """
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

        return config_path

    def create_custom_loss(self):
        """
        Create custom weighted loss function for class imbalance handling.

        Returns:
            nn.Module: Custom weighted loss function
        """
        if self.class_weights_tensor is None:
            print("Warning: No class weights available for custom loss")
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
        """
        Patch the YOLOv12 model to use weighted loss function.

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
        """
        Initialize YOLOv12 model for classification with class weights.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Try classification model first
            model_name = f"yolo12{self.model_size}-cls.pt"
            self.model = YOLO(model_name)
            print(f"YOLOv12{self.model_size} Classification model initialized successfully")

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
                self.model = YOLO(model_name)
                print(f"Using YOLOv12{self.model_size} detection model as fallback")

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
        This replaces both train_model_with_class_weights() and train_model_with_progressive_unfreezing().

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
        print("✓ Automatic checkpoint management between phases")
        print("✓ Layer freezing status tracking")

        print(f"\nTraining summary:")
        print(f"  Total phases: {len(all_phase_results)}")
        print(f"  Total epochs: {epochs}")

        # Return comprehensive results
        return {
            "all_phases": all_phase_results,
            "final_results": all_phase_results[-1]["results"] if all_phase_results else None,
            "class_weights_applied": True,
            "progressive_unfreezing_applied": True,
            "unfreezing_schedule": self.unfreeze_schedule,
            "class_weights": self.class_weights,
        }

    def validate_model(self, classification_path):
        """
        Validate the trained model with comprehensive metrics.

        Args:
            classification_path (str): Path to classification dataset

        Returns:
            dict: Validation results including Top-K metrics
        """
        print("Validating model...")

        # Load best model
        best_model_path = os.path.join(self.save_dir, "weighted_training", "weights", "best.pt")
        if os.path.exists(best_model_path):
            self.model = YOLO(best_model_path)
            print(f"Loaded best model from: {best_model_path}")

        # Run standard validation
        config_path = os.path.join(classification_path, "data.yaml")
        results = self.model.val(data=config_path)

        # Calculate Top-K accuracy metrics
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
        """
        Calculate Top-K accuracy metrics for the model.

        Args:
            classification_path (str): Path to classification dataset
            k_values (list): List of K values for Top-K accuracy

        Returns:
            dict: Top-K accuracy results
        """
        print(f"Calculating Top-K accuracy for k={k_values}...")

        val_path = os.path.join(classification_path, "val")
        y_true = []
        y_pred_probs = []

        # Collect predictions from validation set
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
        """
        Create comprehensive confusion matrix analysis with visualizations.

        Args:
            classification_path (str): Path to classification dataset
        """
        print("Creating confusion matrix analysis...")

        val_path = os.path.join(classification_path, "val")
        y_true = []
        y_pred = []

        # Collect predictions for confusion matrix
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
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Calculate per-class metrics
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
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
        """
        Save comprehensive model information and metadata.
        """
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
                "Stratified train/validation split",
                "Comprehensive confusion matrix analysis",
            ],
        }

        metadata_path = os.path.join(self.save_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model metadata saved to: {metadata_path}")


def main():
    """
    Main training pipeline with unified progressive unfreezing + class weights.
    """
    try:
        # Initialize classifier
        print("Initializing WeightedYOLOv12Classifier...")
        classifier = WeightedYOLOv12Classifier(model_size="n", img_size=640, batch_size=16)

        # Steps 1-3: Dataset download, cleaning, and model initialization
        gdrive_file_id = "1zvCNOz4P0QFdOAfpDpSpidFVjXHFxDUE"
        dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)
        classification_path = classifier.clean_and_extract_objects(
            dataset_path, min_area=0.0001, max_samples_per_class=15000, val_split=0.2
        )

        if not classifier.initialize_yolov12_classifier():
            raise RuntimeError("Failed to initialize YOLOv12 model")

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
        validation_results = classifier.validate_model(classification_path)
        classifier.create_confusion_matrix(classification_path)
        classifier.save_model_info()

        # Cleanup
        if os.path.exists("dataset/"):
            shutil.rmtree("dataset/")

        print("UNIFIED TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        return classifier

    except Exception as e:
        print(f"\nERROR: Training pipeline failed with error: {str(e)}")
        raise


def run_prediction_example(classifier, test_image_path):
    """
    Example function to run predictions on new images.

    Args:
        classifier (WeightedYOLOv12Classifier): Trained classifier instance
        test_image_path (str): Path to test image
    """
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    print(f"\nRunning prediction on: {test_image_path}")

    # Load best model if not already loaded
    best_model_path = os.path.join(classifier.save_dir, "weighted_training", "weights", "best.pt")
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)

        # Run prediction
        results = model.predict(test_image_path, verbose=False)

        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                # Get top predictions
                top_indices = probs.top5
                top_confidences = probs.top5conf.cpu().numpy()

                print("Top 5 predictions:")
                for i, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
                    class_name = classifier.class_names[idx]
                    print(f"  {i+1}. {class_name}: {conf:.3f}")
        else:
            print("No predictions found")
    else:
        print("Trained model not found!")


if __name__ == "__main__":
    # Run the main training pipeline
    trained_classifier = main()
