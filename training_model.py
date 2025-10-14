import datetime
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


class AdaptiveYOLOv12DetectionTrainer:
    """
    YOLOv12 Detection Trainer with aggressive class weighting and intelligent progressive unfreezing.
    Focuses purely on object detection with proper YOLO architecture understanding.
    """

    def __init__(self, model_size="s", img_size=640, batch_size=10, default_conf=0.30):
        """
        Initialize the YOLOv12 Detection Trainer.

        Args:
            model_size (str): Model size ('n', 's', 'm', 'l', 'x')
            img_size (int): Input image size
            batch_size (int): Training batch size
            default_conf (float): Default confidence threshold
        """
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.default_conf = default_conf

        self.model = None
        self.class_names = []
        self.original_class_names = []
        self.class_weights = None
        self.class_weights_tensor = None
        self.id_map = {}
        self.dataset_stats = {}

        # Progressive unfreezing state
        self.unfreezing_state = {
            "initialized": False,
            "current_epoch": 0,
            "current_phase": 0,
            "layer_groups": [],
            "schedule": None,
        }

        # Create timestamped save directory
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/YOLOv12_Detection_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory: {self.save_dir}")

    def download_and_extract_dataset(self, gdrive_file_id, output_filename=None):
        """Download and extract dataset from Google Drive."""
        if output_filename is None:
            output_filename = f"dataset_{gdrive_file_id}.zip"

        url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, output_filename)
            print("Descargando dataset desde Google Drive...")
            gdown.download(url, temp_zip_path, quiet=False)

            extract_path = "dataset/"
            print("Extrayendo dataset...")
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Buscar data.yaml en toda la jerarquía
            yaml_path = None
            for root, dirs, files in os.walk(extract_path):
                if "data.yaml" in files:
                    yaml_path = os.path.join(root, "data.yaml")
                    break

            if yaml_path is None:
                raise FileNotFoundError("No se encontró 'data.yaml' dentro del dataset extraído.")

            # Ajustar extract_path para que sea la carpeta que contiene data.yaml
            extract_path = os.path.dirname(yaml_path)

            print(f"Dataset extraído correctamente. Archivo 'data.yaml' encontrado en: {yaml_path}")
            return extract_path

    def prepare_detection_dataset(self, dataset_path, min_area=0.0, val_split=0.2):
        """
        Prepare YOLO detection dataset with proper structure and aggressive class weighting.

        Args:
            dataset_path (str): Path to raw dataset
            min_area (float): Minimum normalized area for valid boxes
            val_split (float): Validation split ratio

        Returns:
            str: Path to prepared dataset
        """
        print("Preparing YOLO detection dataset...")

        # Load original dataset configuration
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset config not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data_config = yaml.safe_load(f)
            self.original_class_names = data_config.get("names", [])

        print(f"Original classes: {len(self.original_class_names)}")

        # Filter background classes
        filtered_classes, id_map = self._filter_background_classes()

        if len(filtered_classes) == 0:
            raise ValueError("No valid classes after background filtering!")

        self.class_names = filtered_classes
        self.id_map = id_map

        print(f"Filtered classes ({len(self.class_names)}): {self.class_names}")

        # Setup detection dataset structure
        detection_dir = "yolo_detection_dataset"
        if os.path.exists(detection_dir):
            shutil.rmtree(detection_dir)

        self._create_detection_structure(detection_dir)

        # Process train/val splits
        original_train_dir = os.path.join(dataset_path, "train")

        if not os.path.exists(original_train_dir):
            raise FileNotFoundError(f"Training directory not found: {original_train_dir}")

        # Get all images and create stratified split
        all_images = self._get_valid_images(original_train_dir)
        train_imgs, val_imgs = self._create_detection_split(
            all_images, original_train_dir, val_split
        )

        # Process splits
        train_stats = self._process_detection_split(
            train_imgs, original_train_dir, detection_dir, "train", min_area
        )
        val_stats = self._process_detection_split(
            val_imgs, original_train_dir, detection_dir, "val", min_area
        )

        # Calculate aggressive class weights
        self._calculate_aggressive_detection_weights(train_stats, val_stats)

        # Create YOLO detection config
        config_path = self._create_detection_config(detection_dir)

        print(f"\nDataset preparation complete:")
        print(f"  Train: {len(train_imgs)} images")
        print(f"  Val: {len(val_imgs)} images")
        print(f"  Classes: {len(self.class_names)}")

        return config_path

    def _filter_background_classes(self):
        """Filter out background classes and create ID mapping."""
        background_keywords = ["background", "bg", "__background__", "void", "unlabeled", "unknown"]

        filtered_classes = []
        id_map = {}
        new_class_id = 0

        for original_id, class_name in enumerate(self.original_class_names):
            clean_name = class_name.strip().lower()
            if clean_name not in background_keywords:
                filtered_classes.append(class_name.strip())
                id_map[original_id] = new_class_id
                new_class_id += 1

        return filtered_classes, id_map

    def _create_detection_structure(self, detection_dir):
        """Create proper YOLO detection directory structure."""
        structure_dirs = [
            os.path.join(detection_dir, "images", "train"),
            os.path.join(detection_dir, "images", "val"),
            os.path.join(detection_dir, "labels", "train"),
            os.path.join(detection_dir, "labels", "val"),
        ]

        for dir_path in structure_dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _get_valid_images(self, train_dir):
        """Get list of valid images with corresponding labels."""
        images_dir = os.path.join(train_dir, "images")
        labels_dir = os.path.join(train_dir, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError("Train images or labels directory not found")

        all_images = []
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_file)

                if os.path.exists(label_path):
                    all_images.append(img_file)

        return all_images

    def _create_detection_split(self, all_images, train_dir, val_split):
        """Create train/val split for detection dataset."""
        labels_dir = os.path.join(train_dir, "labels")

        # Try stratified split based on primary class per image
        try:
            image_primary_classes = []
            valid_images = []

            for img_file in all_images:
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_file)

                primary_class = self._get_primary_class_from_label(label_path)
                if primary_class is not None:
                    image_primary_classes.append(primary_class)
                    valid_images.append(img_file)

            if len(set(image_primary_classes)) > 1:
                train_imgs, val_imgs = train_test_split(
                    valid_images,
                    test_size=val_split,
                    stratify=image_primary_classes,
                    random_state=42,
                )
                return train_imgs, val_imgs
        except Exception as e:
            print(f"Stratified split failed: {e}. Using random split.")

        # Fallback to random split
        train_imgs, val_imgs = train_test_split(all_images, test_size=val_split, random_state=42)
        return train_imgs, val_imgs

    def _get_primary_class_from_label(self, label_path):
        """Get primary class (most frequent) from label file."""
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            classes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in self.id_map:
                        classes.append(self.id_map[class_id])

            if classes:
                return max(set(classes), key=classes.count)
        except:
            pass
        return None

    def _process_detection_split(self, images, source_dir, target_dir, split_name, min_area):
        """Process images and labels for detection dataset."""
        source_imgs_dir = os.path.join(source_dir, "images")
        source_labels_dir = os.path.join(source_dir, "labels")

        target_imgs_dir = os.path.join(target_dir, "images", split_name)
        target_labels_dir = os.path.join(target_dir, "labels", split_name)

        class_counts = {name: 0 for name in self.class_names}
        total_boxes, kept_boxes = 0, 0
        processed_images = 0

        print(f"Processing {split_name} with min_area: {min_area}")
        print(f"id_map covers classes: {list(self.id_map.keys())}")

        # Procesa solo los primeros 10 archivos para debug
        debug_images = images[:10] if len(images) > 10 else images
        sample_stats = []

        for img_file in debug_images:
            src_label_path = os.path.join(source_labels_dir, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(src_label_path):
                with open(src_label_path, "r") as f:
                    lines = f.readlines()
                    sample_stats.append(len(lines))

        print(
            f"Sample - Avg boxes per image: {sum(sample_stats)/len(sample_stats) if sample_stats else 0}"
        )

        for img_file in images:
            # Copy image
            src_img_path = os.path.join(source_imgs_dir, img_file)
            dst_img_path = os.path.join(target_imgs_dir, img_file)

            if not os.path.exists(src_img_path):
                continue

            # Process labels
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label_path = os.path.join(source_labels_dir, label_file)
            dst_label_path = os.path.join(target_labels_dir, label_file)

            if os.path.exists(src_label_path):
                processed_labels = self._process_detection_labels(src_label_path, min_area)

                if processed_labels["kept_lines"]:
                    shutil.copy2(src_img_path, dst_img_path)

                    with open(dst_label_path, "w") as f:
                        f.writelines(processed_labels["kept_lines"])

                    processed_images += 1
                    total_boxes += processed_labels["total_boxes"]
                    kept_boxes += processed_labels["kept_boxes"]

                    for class_name, count in processed_labels["class_counts"].items():
                        class_counts[class_name] += count

        stats = {
            "images": processed_images,
            "total_boxes": total_boxes,
            "kept_boxes": kept_boxes,
            "class_counts": class_counts,
        }

        print(
            f"{split_name.capitalize()} split: {processed_images} images, "
            f"{kept_boxes}/{total_boxes} boxes kept"
        )

        return stats

    def _process_detection_labels(self, label_path, min_area):
        """Process single label file for detection."""
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
        except:
            return {
                "kept_lines": [],
                "total_boxes": 0,
                "kept_boxes": 0,
                "class_counts": {name: 0 for name in self.class_names},
            }

        kept_lines = []
        class_counts = {name: 0 for name in self.class_names}
        total_boxes = len(lines)

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                continue

            # Validate bounding box
            if (
                class_id not in self.id_map
                or w <= 0
                or h <= 0
                or cx < 0
                or cy < 0
                or cx > 1
                or cy > 1
                or w * h < min_area
            ):
                continue

            # Remap class ID
            new_class_id = self.id_map[class_id]
            kept_lines.append(f"{new_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            if new_class_id < len(self.class_names):
                class_counts[self.class_names[new_class_id]] += 1

        return {
            "kept_lines": kept_lines,
            "total_boxes": total_boxes,
            "kept_boxes": len(kept_lines),
            "class_counts": class_counts,
        }

    def _calculate_aggressive_detection_weights(self, train_stats, val_stats):
        """
        Calculate aggressive class weights for severe imbalance correction in detection.
        Uses your original aggressive weighting logic adapted for detection.
        """
        print("\nCalculating aggressive class weights for detection...")

        # Combine train and val statistics
        combined_counts = {}
        for class_name in self.class_names:
            combined_counts[class_name] = (
                train_stats["class_counts"][class_name] + val_stats["class_counts"][class_name]
            )

        valid_classes = {cls: count for cls, count in combined_counts.items() if count > 0}

        if len(valid_classes) == 0:
            print("Warning: No valid classes with samples")
            return

        counts = np.array([valid_classes[name] for name in valid_classes.keys()])
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count

        print(f"Detection dataset imbalance ratio: {imbalance_ratio:.2f}")

        # Apply aggressive weighting strategy based on imbalance severity
        if imbalance_ratio > 40:
            # Logarithmic weighting for extreme cases
            log_weights = np.log(max_count + 1) / np.log(counts + 1)
            aggressive_weights = log_weights * 3.0  # High amplification
            strategy = "logarithmic extreme (3x amplified)"

        elif imbalance_ratio > 20:
            # Power weighting for severe cases
            power_weights = np.power(max_count / counts, 0.75)
            aggressive_weights = power_weights * 2.0
            strategy = "power weighting (0.75 exp, 2x amplified)"

        elif imbalance_ratio > 10:
            # Enhanced square root for high imbalance
            sqrt_weights = np.sqrt(max_count / counts)
            aggressive_weights = sqrt_weights * 2.5
            strategy = "enhanced sqrt (2.5x amplified)"

        else:
            # Amplified linear for moderate imbalance
            linear_weights = max_count / counts
            aggressive_weights = linear_weights * 1.2
            strategy = "amplified linear (1.2x)"

        print(f"Applied {strategy}")

        # Boost very rare classes (< 1% of max class)
        rare_threshold = max_count * 0.05
        for i, count in enumerate(counts):
            if count < rare_threshold:
                aggressive_weights[i] *= 4.0  # 4x boost for very rare classes
                print(
                    f"Rare class boost applied: {list(valid_classes.keys())[i]} "
                    f"({count} samples)"
                )

        # Clip to reasonable range for detection (higher than classification)
        aggressive_weights = np.clip(aggressive_weights, 0.1, 100.0)

        # Optional smoothing for extreme variance
        weight_std = np.std(aggressive_weights)
        weight_mean = np.mean(aggressive_weights)
        if len(aggressive_weights) > 2 and weight_std > weight_mean * 0.8:
            # Gentle smoothing
            smoothed = np.copy(aggressive_weights)
            for i in range(1, len(smoothed) - 1):
                smoothed[i] = 0.7 * aggressive_weights[i] + 0.15 * (
                    aggressive_weights[i - 1] + aggressive_weights[i + 1]
                )
            aggressive_weights = smoothed
            print("Applied weight smoothing for stability")

        # Store weights
        self.class_weights = {}
        default_weight = np.mean(aggressive_weights) if len(aggressive_weights) > 0 else 1.0
        valid_class_names = list(valid_classes.keys())

        for i, class_name in enumerate(self.class_names):
            if class_name in valid_class_names:
                weight_idx = valid_class_names.index(class_name)
                self.class_weights[i] = float(aggressive_weights[weight_idx])
            else:
                self.class_weights[i] = default_weight

        # Create tensor for loss function
        weight_values = [self.class_weights[i] for i in range(len(self.class_names))]
        self.class_weights_tensor = torch.FloatTensor(weight_values)

        # Detailed reporting
        print("\nDetection class weights:")
        sorted_weights = sorted(enumerate(weight_values), key=lambda x: x[1], reverse=True)

        total_samples = sum(combined_counts.values())
        for i, weight in sorted_weights:
            if i < len(self.class_names):
                class_name = self.class_names[i]
                sample_count = combined_counts.get(class_name, 0)
                percentage = (sample_count / total_samples) * 100 if total_samples > 0 else 0
                print(
                    f"  {class_name}: weight={weight:.3f} "
                    f"(samples: {sample_count}, {percentage:.1f}%)"
                )

        # Weight statistics
        weight_ratio = max(weight_values) / min(weight_values)
        print(f"\nWeight statistics:")
        print(f"  Max/Min ratio: {weight_ratio:.1f}")
        print(f"  Mean weight: {np.mean(weight_values):.3f}")
        print(f"  Weight std: {np.std(weight_values):.3f}")

    def _create_detection_config(self, detection_dir):
        """Create YOLO detection configuration file."""
        config = {
            "path": os.path.abspath(detection_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        config_path = os.path.join(detection_dir, "data.yaml")

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        print(f"YOLO detection config saved: {config_path}")
        return config_path

    def initialize_yolo_detection_model(self):
        """Initialize YOLOv12 detection model and apply class weights."""
        if len(self.class_names) == 0:
            print("Error: No classes defined")
            return False

        try:
            model_name = f"yolo12{self.model_size}.pt"
            print(f"Initializing {model_name}...")
            self.model = YOLO(model_name)

            print(f"YOLOv12{self.model_size} detection model loaded successfully")

            # Apply class weights if available
            if self.class_weights_tensor is not None:
                self._apply_detection_class_weights()

            return True

        except Exception as e:
            print(f"Error initializing model: {e}")
            return False

    def _apply_detection_class_weights(self):
        """Apply class weights to YOLO detection model."""
        if self.model is None or self.class_weights_tensor is None:
            return False

        try:
            # Get device from model
            device = next(self.model.model.parameters()).device
            weighted_tensor = self.class_weights_tensor.to(device)

            # Store original loss function
            if hasattr(self.model.model, "loss"):
                original_loss = self.model.model.loss

                def weighted_detection_loss(*args, **kwargs):
                    """Modified loss function with aggressive class weights for detection."""
                    loss_dict = original_loss(*args, **kwargs)

                    # Apply weights to classification loss component
                    if "cls" in loss_dict and len(args) >= 2:
                        try:
                            preds, targets = args[0], args[1]

                            # Handle different target formats
                            if hasattr(targets, "cls"):
                                cls_targets = targets.cls.long()
                            elif isinstance(targets, dict) and "cls" in targets:
                                cls_targets = targets["cls"].long()
                            else:
                                return loss_dict  # Skip if can't extract targets

                            # Handle different prediction formats
                            if isinstance(preds, (list, tuple)):
                                cls_preds = preds[0] if len(preds) > 0 else None
                            else:
                                cls_preds = preds

                            if cls_preds is not None and cls_preds.size(1) == len(self.class_names):
                                # Apply weighted focal loss for better recall
                                ce_loss = nn.CrossEntropyLoss(
                                    weight=weighted_tensor, reduction="none"
                                )(cls_preds, cls_targets)

                                # Focal loss with alpha=4 for recall boost
                                pt = torch.exp(-ce_loss)
                                focal_loss = (1 - pt) ** 4 * ce_loss

                                loss_dict["cls"] = focal_loss.mean()

                        except Exception as e:
                            print(f"Warning: Could not apply class weights: {e}")

                    return loss_dict

                # Replace loss function
                self.model.model.loss = weighted_detection_loss
                print("Applied aggressive class weights with focal loss to detection model")
                return True

        except Exception as e:
            print(f"Error applying class weights: {e}")

        return False

    def setup_detection_progressive_unfreezing(self, unfreeze_schedule=None):
        """
        Setup progressive unfreezing optimized for YOLO detection architecture.

        Args:
            unfreeze_schedule (dict): Custom unfreezing schedule

        Returns:
            dict: Unfreezing configuration
        """
        if self.model is None:
            print("Error: Model not initialized")
            return None

        if unfreeze_schedule is None:
            # Detection-optimized unfreezing schedule
            {
                0: 0.20,  # Detection heads only (0-30 epochs)
                30: 0.40,  # + Neck PAN (30-60 epochs)
                60: 0.60,  # + Neck FPN (60-100 epochs)
                100: 0.75,  # + Late backbone (100-140 epochs)
                140: 0.90,  # + Mid backbone (140-180 epochs)
                180: 1.0,  # Full model (180-210 epochs)
            }

        print("Setting up YOLO detection progressive unfreezing...")

        # Get trainable parameters
        param_info = []

        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                param_info.append(
                    {
                        "name": name,
                        "param": param,
                        "shape": tuple(param.shape),
                        "numel": param.numel(),
                    }
                )

        # Create YOLO-specific layer groups
        layer_groups = self._create_detection_layer_groups(param_info)

        print(f"Created {len(layer_groups)} detection layer groups:")
        for i, group in enumerate(layer_groups):
            print(f"  Group {i+1}: {group['name']} ({len(group['params'])} parameters)")

        # Store unfreezing state
        self.unfreezing_state = {
            "initialized": True,
            "schedule": unfreeze_schedule,
            "layer_groups": layer_groups,
            "total_params": len(param_info),
            "current_epoch": 0,
            "current_phase": 0,
        }

        # Apply initial freezing
        self._apply_detection_unfreezing_phase(0)

        return {
            "schedule": unfreeze_schedule,
            "total_params": len(param_info),
            "layer_groups": len(layer_groups),
            "initial_phase": 0,
        }

    def _create_detection_layer_groups(self, param_info):
        """Create YOLO detection-specific layer groups for unfreezing."""
        groups = []

        # YOLO detection layer categorization
        detection_heads = []
        neck_layers = []
        late_backbone = []
        mid_backbone = []
        early_backbone = []

        for info in param_info:
            name = info["name"].lower()

            # Detection heads (highest priority - most task-specific)
            if any(x in name for x in ["detect", "cv2", "cv3", "dfl", "head"]):
                detection_heads.append(info)

            # Neck/FPN layers (second priority - multi-scale aggregation)
            elif any(x in name for x in ["neck", "fpn", "pan", "sppf", "concat", "upsample"]):
                neck_layers.append(info)

            # Backbone layers (by depth - deeper = higher level features)
            elif "model" in name:
                # Extract layer number for backbone ordering
                import re

                layer_match = re.search(r"model\.(\d+)", name)
                if layer_match:
                    layer_num = int(layer_match.group(1))

                    if layer_num >= 15:  # Late backbone (high-level features)
                        late_backbone.append(info)
                    elif layer_num >= 9:  # Mid backbone
                        mid_backbone.append(info)
                    else:  # Early backbone (low-level features)
                        early_backbone.append(info)
                else:
                    # Default to mid backbone if can't determine layer
                    mid_backbone.append(info)
            else:
                # Other parameters go to mid backbone
                mid_backbone.append(info)

        # Create groups in unfreezing priority order
        if detection_heads:
            groups.append(
                {
                    "name": "Detection Heads (cv2, cv3, dfl)",
                    "params": detection_heads,
                    "priority": 1,
                }
            )

        if neck_layers:
            groups.append(
                {"name": "Neck/FPN (SPPF, Concat, Upsample)", "params": neck_layers, "priority": 2}
            )

        if late_backbone:
            groups.append(
                {
                    "name": "Backbone Late (High-level features)",
                    "params": late_backbone,
                    "priority": 3,
                }
            )

        if mid_backbone:
            groups.append(
                {"name": "Backbone Mid (Mid-level features)", "params": mid_backbone, "priority": 4}
            )

        if early_backbone:
            groups.append(
                {
                    "name": "Backbone Early (Low-level features)",
                    "params": early_backbone,
                    "priority": 5,
                }
            )

        # Sort by priority
        groups.sort(key=lambda x: x["priority"])

        return groups

    def _apply_detection_unfreezing_phase(self, epoch):
        """Apply progressive unfreezing for current epoch."""
        if not self.unfreezing_state["initialized"]:
            return False

        schedule = self.unfreezing_state["schedule"]
        layer_groups = self.unfreezing_state["layer_groups"]

        # Find current phase ratio
        current_phase_ratio = 0.15  # Default
        for epoch_threshold in sorted(schedule.keys()):
            if epoch >= epoch_threshold:
                current_phase_ratio = schedule[epoch_threshold]
            else:
                break

        # Calculate groups to unfreeze
        total_groups = len(layer_groups)
        groups_to_unfreeze = max(1, int(total_groups * current_phase_ratio))
        groups_to_unfreeze = min(groups_to_unfreeze, total_groups)

        old_phase = self.unfreezing_state.get("current_phase", 0)

        if groups_to_unfreeze != old_phase or epoch == 0:
            print(f"\nEpoch {epoch}: Detection unfreezing phase")
            print(
                f"  Unfreezing {groups_to_unfreeze}/{total_groups} groups "
                f"({current_phase_ratio*100:.0f}%)"
            )

            # Freeze all parameters first
            for group in layer_groups:
                for param_info in group["params"]:
                    param_info["param"].requires_grad = False

            # Unfreeze specified groups (by priority)
            unfrozen_params = 0
            for i in range(groups_to_unfreeze):
                group = layer_groups[i]
                for param_info in group["params"]:
                    param_info["param"].requires_grad = True
                    unfrozen_params += 1
                print(f"    ✓ {group['name']} ({len(group['params'])} params)")

            # Update state
            self.unfreezing_state["current_phase"] = groups_to_unfreeze
            self.unfreezing_state["current_epoch"] = epoch

            frozen_params = self.unfreezing_state["total_params"] - unfrozen_params
            print(f"  Total: {unfrozen_params} unfrozen, {frozen_params} frozen")

            return True

        return False

    def train_detection_model_with_progressive_unfreezing(
        self, config_path, epochs=210, unfreeze_schedule=None
    ):
        """
        Train YOLO detection model with progressive unfreezing and aggressive class weights.

        Args:
            config_path (str): Path to YOLO detection config
            epochs (int): Total training epochs
            unfreeze_schedule (dict): Custom unfreezing schedule

        Returns:
            dict: Training results
        """
        print(f"\n{'='*60}")
        print("YOLO DETECTION TRAINING: PROGRESSIVE UNFREEZING + AGGRESSIVE WEIGHTS")
        print(f"{'='*60}")

        if len(self.class_names) == 0:
            raise ValueError("No classes defined")

        if self.model is None:
            raise ValueError("Model not initialized")

        # Setup progressive unfreezing
        unfreezing_config = self.setup_detection_progressive_unfreezing(unfreeze_schedule)
        if unfreezing_config is None:
            print("Failed to setup progressive unfreezing")
            return None

        print("Training features enabled:")
        print(f"  ✓ Progressive unfreezing ({len(self.unfreezing_state['layer_groups'])} groups)")
        print(
            f"  ✓ Aggressive class weights (max ratio: {max(self.class_weights.values())/min(self.class_weights.values()):.1f})"
        )
        print(f"  ✓ Detection-optimized architecture understanding")

        # Base training arguments for detection
        base_training_args = {
            "data": config_path,
            "imgsz": self.img_size,
            "batch": self.batch_size,
            "device": "cpu",
            "workers": 4,
            "patience": 20,
            "save": True,
            "save_period": 15,
            "val": True,
            "project": self.save_dir,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "SGD",
            "lr0": 0.005,
            "lrf": 0.0001,
            "momentum": 0.95,
            "weight_decay": 0.0005,
            "warmup_epochs": 8,
            "warmup_momentum": 0.85,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,
            "verbose": True,
            "dropout": 0.1,
            "label_smoothing": 0.05,
            "conf": self.default_conf,
            "iou": 0.5,  # Detection-specific
            "close_mosaic": 10,
        }

        # Phase-based training
        schedule = self.unfreezing_state["schedule"]
        epoch_phases = sorted(schedule.keys())
        all_phase_results = []

        print("\nStarting detection training with progressive unfreezing...")

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
                f"DETECTION PHASE {i+1}: EPOCHS {phase_start_epoch}-{phase_start_epoch + phase_epochs - 1}"
            )
            print(f"{'='*50}")

            # Apply unfreezing for this phase
            self._apply_detection_unfreezing_phase(phase_start_epoch)

            # Load previous checkpoint if continuing
            if i > 0:
                previous_phase_name = f"detection_phase_{i}"
                last_checkpoint = os.path.join(
                    self.save_dir, previous_phase_name, "weights", "last.pt"
                )
                if os.path.exists(last_checkpoint):
                    print(f"Loading checkpoint: {last_checkpoint}")
                    self.model = YOLO(last_checkpoint)
                    # Re-apply class weights and unfreezing
                    self._apply_detection_class_weights()
                    self._apply_detection_unfreezing_phase(phase_start_epoch)

            # Configure training for this phase
            phase_training_args = base_training_args.copy()
            phase_training_args.update(
                {"epochs": phase_epochs, "name": f"detection_phase_{i+1}", "conf": 0.3}
            )

            # Execute training
            print(f"Starting detection training phase {i+1}...")
            try:
                phase_results = self.model.train(**phase_training_args)
                all_phase_results.append(
                    {
                        "phase": i + 1,
                        "start_epoch": phase_start_epoch,
                        "epochs": phase_epochs,
                        "results": phase_results,
                    }
                )
                print(f"Detection phase {i+1} completed successfully!")

            except Exception as e:
                print(f"Error in detection phase {i+1}: {e}")
                continue

        # Save training summary
        self._save_detection_training_summary(all_phase_results)

        print(f"\n{'='*60}")
        print("DETECTION TRAINING COMPLETED!")
        print(f"{'='*60}")

        return {
            "all_phases": all_phase_results,
            "final_results": all_phase_results[-1]["results"] if all_phase_results else None,
            "progressive_unfreezing": True,
            "aggressive_class_weights": True,
            "detection_optimized": True,
            "progressive_lr": True,
            "class_weights": self.class_weights,
            "final_classes": self.class_names,
            "final_confidence": self.default_conf,
        }

    def _save_detection_training_summary(self, all_phase_results):
        """Save comprehensive training summary."""
        summary = {
            "training_timestamp": datetime.datetime.now().isoformat(),
            "model_config": {
                "model_size": self.model_size,
                "img_size": self.img_size,
                "batch_size": self.batch_size,
            },
            "dataset_info": {
                "classes": self.class_names,
                "num_classes": len(self.class_names),
                "class_weights": self.class_weights,
            },
            "training_phases": len(all_phase_results),
            "progressive_unfreezing": {
                "enabled": True,
                "schedule": self.unfreezing_state["schedule"],
                "layer_groups": len(self.unfreezing_state["layer_groups"]),
            },
            "phase_results": all_phase_results,
        }

        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Training summary saved: {summary_path}")

    def validate_all_checkpoints(self, config_path):
        """
        Validate all checkpoints (best.pt and last.pt) from all training phases.
        Find and return the best overall model.

        Args:
            config_path (str): Path to YOLO detection config

        Returns:
            dict: Comprehensive validation results
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE CHECKPOINT VALIDATION")
        print("=" * 70)

        all_results = []

        # Iterate through all phases
        phase_dirs = sorted(
            [d for d in os.listdir(self.save_dir) if d.startswith("detection_phase_")]
        )

        for phase_dir in phase_dirs:
            phase_num = int(phase_dir.split("_")[-1])
            phase_path = os.path.join(self.save_dir, phase_dir)
            weights_dir = os.path.join(phase_path, "weights")

            if not os.path.exists(weights_dir):
                continue

            # Validate best.pt
            best_path = os.path.join(weights_dir, "best.pt")
            if os.path.exists(best_path):
                print(f"\nValidating Phase {phase_num} - best.pt...")
                try:
                    model = YOLO(best_path)
                    results = model.val(data=config_path, conf=0.3, iou=0.5, verbose=False)

                    result_data = {
                        "phase": phase_num,
                        "checkpoint": "best.pt",
                        "path": best_path,
                        "mAP50": float(results.box.map50),
                        "mAP50-95": float(results.box.map),
                        "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                        "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                    }

                    # Extract class-wise results
                    class_results = self._extract_class_results(results)
                    result_data["class_results"] = class_results

                    all_results.append(result_data)
                    print(
                        f"  mAP50: {result_data['mAP50']:.3f} | mAP50-95: {result_data['mAP50-95']:.3f}"
                    )

                except Exception as e:
                    print(f"  Error validating best.pt: {e}")

            # Validate last.pt
            last_path = os.path.join(weights_dir, "last.pt")
            if os.path.exists(last_path):
                print(f"Validating Phase {phase_num} - last.pt...")
                try:
                    model = YOLO(last_path)
                    results = model.val(data=config_path, conf=0.3, iou=0.5, verbose=False)

                    result_data = {
                        "phase": phase_num,
                        "checkpoint": "last.pt",
                        "path": last_path,
                        "mAP50": float(results.box.map50),
                        "mAP50-95": float(results.box.map),
                        "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                        "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                    }

                    # Extract class-wise results
                    class_results = self._extract_class_results(results)
                    result_data["class_results"] = class_results

                    all_results.append(result_data)
                    print(
                        f"  mAP50: {result_data['mAP50']:.3f} | mAP50-95: {result_data['mAP50-95']:.3f}"
                    )

                except Exception as e:
                    print(f"  Error validating last.pt: {e}")

        if not all_results:
            print("No checkpoints found to validate!")
            return None

        # Find best overall model
        best_overall = max(all_results, key=lambda x: x["mAP50-95"])

        print("\n" + "=" * 70)
        print("CHECKPOINT COMPARISON SUMMARY")
        print("=" * 70)
        print(
            f"{'Phase':<8} {'Checkpoint':<12} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}"
        )
        print("-" * 70)

        for result in all_results:
            marker = " 🏆" if result == best_overall else ""
            print(
                f"{result['phase']:<8} {result['checkpoint']:<12} "
                f"{result['mAP50']:<10.3f} {result['mAP50-95']:<10.3f} "
                f"{result['precision']:<10.3f} {result['recall']:<10.3f}{marker}"
            )

        print("\n" + "=" * 70)
        print("BEST OVERALL MODEL")
        print("=" * 70)
        print(f"Phase:        {best_overall['phase']}")
        print(f"Checkpoint:   {best_overall['checkpoint']}")
        print(f"Path:         {best_overall['path']}")
        print(f"mAP50:        {best_overall['mAP50']:.3f}")
        print(f"mAP50-95:     {best_overall['mAP50-95']:.3f}")
        print(f"Precision:    {best_overall['precision']:.3f}")
        print(f"Recall:       {best_overall['recall']:.3f}")

        # Print class-wise results for best model
        if best_overall["class_results"]:
            print("\n" + "=" * 70)
            print("BEST MODEL - CLASS-WISE RESULTS")
            print("=" * 70)
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
            print("-" * 70)

            for class_result in best_overall["class_results"]:
                print(
                    f"{class_result['class']:<20} "
                    f"{class_result['precision']:<12.3f} "
                    f"{class_result['recall']:<12.3f} "
                    f"{class_result['mAP50']:<12.3f} "
                    f"{class_result['mAP50-95']:<12.3f}"
                )

        # Save comprehensive results
        self._save_validation_comparison(all_results, best_overall)

        # Copy best model to main directory
        best_model_dest = os.path.join(self.save_dir, "best_overall_model.pt")
        shutil.copy2(best_overall["path"], best_model_dest)
        print(f"\nBest model copied to: {best_model_dest}")

        return {
            "all_results": all_results,
            "best_overall": best_overall,
            "best_model_path": best_model_dest,
        }

    def _extract_class_results(self, results):
        """
        Extract class-wise metrics from validation results.

        Args:
            results: YOLO validation results object

        Returns:
            list: List of dicts with class-wise metrics
        """
        class_results = []

        if hasattr(results, "box") and hasattr(results.box, "ap_class_index"):
            for idx, class_idx in enumerate(results.box.ap_class_index):
                if class_idx < len(self.class_names):
                    class_data = {
                        "class": self.class_names[class_idx],
                        "class_id": int(class_idx),
                        "precision": (
                            float(results.box.p[idx])
                            if hasattr(results.box, "p") and idx < len(results.box.p)
                            else 0.0
                        ),
                        "recall": (
                            float(results.box.r[idx])
                            if hasattr(results.box, "r") and idx < len(results.box.r)
                            else 0.0
                        ),
                        "mAP50": (
                            float(results.box.ap50[idx]) if idx < len(results.box.ap50) else 0.0
                        ),
                        "mAP50-95": (
                            float(results.box.ap[idx]) if idx < len(results.box.ap) else 0.0
                        ),
                    }
                    class_results.append(class_data)

        return class_results

    def aggressive_minority_oversampling(
        self, dataset_dir, target_samples_per_class=5000, minority_threshold=2500
    ):
        """
        Aggressive oversampling of minority classes with strong augmentations.

        Args:
            dataset_dir (str): Path to cleaned detection dataset
            target_samples_per_class (int): Target number of samples for minority classes
            minority_threshold (int): Classes below this are considered minority

        Returns:
            str: Path to oversampled dataset
        """
        print("\n" + "=" * 70)
        print("AGGRESSIVE MINORITY CLASS OVERSAMPLING")
        print("=" * 70)

        # Analyze current class distribution
        train_images_dir = os.path.join(dataset_dir, "images", "train")
        train_labels_dir = os.path.join(dataset_dir, "labels", "train")

        # Count samples per class
        class_samples = {name: 0 for name in self.class_names}
        images_by_class = {name: [] for name in self.class_names}

        for img_file in os.listdir(train_images_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(train_labels_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()

                # Track which classes are in this image
                classes_in_image = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                            class_samples[class_name] += 1
                            classes_in_image.add(class_name)

                # Add image to all classes it contains
                for class_name in classes_in_image:
                    images_by_class[class_name].append(img_file)

        # Display current distribution
        print("\nCurrent class distribution:")
        print(f"{'Class':<20} {'Samples':<10} {'Images':<10} {'Status':<15}")
        print("-" * 70)

        minority_classes = []
        for class_name in self.class_names:
            samples = class_samples[class_name]
            images = len(images_by_class[class_name])
            status = "MINORITY" if samples < minority_threshold else "OK"

            print(f"{class_name:<20} {samples:<10} {images:<10} {status:<15}")

            if samples < minority_threshold and samples > 0:
                minority_classes.append(class_name)

        if not minority_classes:
            print("\nNo minority classes found. Skipping oversampling.")
            return dataset_dir

        print(f"\nMinority classes to oversample: {minority_classes}")

        # Calculate replication factors
        replication_plan = {}
        for class_name in minority_classes:
            current_samples = class_samples[class_name]
            target = min(target_samples_per_class, current_samples * 10)  # Cap at 10x
            replication_factor = max(2, int(target / current_samples))
            replication_plan[class_name] = {
                "current": current_samples,
                "target": target,
                "factor": replication_factor,
                "images": images_by_class[class_name],
            }

        print("\nOversampling plan:")
        for class_name, plan in replication_plan.items():
            print(
                f"  {class_name}: {plan['current']} → ~{plan['target']} "
                f"({plan['factor']}x, {len(plan['images'])} images)"
            )

        # Create oversampled dataset
        oversampled_dir = dataset_dir + "_oversampled"
        if os.path.exists(oversampled_dir):
            print(f"\nRemoving existing oversampled directory: {oversampled_dir}")
            shutil.rmtree(oversampled_dir)

        # Copy original dataset structure
        print("\nCopying original dataset...")
        shutil.copytree(dataset_dir, oversampled_dir)

        oversampled_images_dir = os.path.join(oversampled_dir, "images", "train")
        oversampled_labels_dir = os.path.join(oversampled_dir, "labels", "train")

        # Apply oversampling with augmentations
        print("\nApplying oversampling with strong augmentations...")

        total_augmented = 0
        for class_name, plan in replication_plan.items():
            print(f"\nProcessing {class_name}...")
            class_augmented = 0

            for img_file in plan["images"]:
                img_path = os.path.join(train_images_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(train_labels_dir, label_file)

                # Create (factor - 1) augmented versions (original already copied)
                for aug_idx in range(plan["factor"] - 1):
                    try:
                        # Apply strong augmentation
                        aug_img, aug_labels = self._apply_strong_augmentation(img_path, label_path)

                        # Save augmented image
                        base_name = os.path.splitext(img_file)[0]
                        aug_img_name = f"{base_name}_aug_{class_name}_{aug_idx}.jpg"
                        aug_img_path = os.path.join(oversampled_images_dir, aug_img_name)

                        cv2.imwrite(aug_img_path, aug_img)

                        # Save augmented labels
                        aug_label_name = f"{base_name}_aug_{class_name}_{aug_idx}.txt"
                        aug_label_path = os.path.join(oversampled_labels_dir, aug_label_name)

                        with open(aug_label_path, "w") as f:
                            f.writelines(aug_labels)

                        class_augmented += 1
                        total_augmented += 1

                    except Exception as e:
                        print(f"  Warning: Failed to augment {img_file}: {e}")
                        continue

            print(f"  Created {class_augmented} augmented images for {class_name}")

        print(f"\nOversampling complete!")
        print(f"   Total augmented images: {total_augmented}")
        print(f"   Oversampled dataset: {oversampled_dir}")

        # Update dataset config
        config_path = os.path.join(oversampled_dir, "data.yaml")
        config = {
            "path": os.path.abspath(oversampled_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        print(f"   Updated config: {config_path}")

        # Generate oversampling report
        self._generate_oversampling_report(
            dataset_dir, oversampled_dir, replication_plan, total_augmented
        )

        return oversampled_dir

    def _apply_strong_augmentation(self, img_path, label_path):
        """
        Apply strong augmentations including flips and rotations.

        Args:
            img_path (str): Path to original image
            label_path (str): Path to original label file

        Returns:
            tuple: (augmented_image, augmented_labels)
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        h, w = img.shape[:2]

        # Read labels
        with open(label_path, "r") as f:
            labels = f.readlines()

        # Parse labels
        boxes = []
        for line in labels:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append([class_id, cx, cy, bw, bh])

        # Randomly select augmentation
        aug_type = np.random.choice(
            ["horizontal_flip", "vertical_flip", "rotate_90_cw", "rotate_90_ccw", "rotate_180"]
        )

        # Apply augmentation
        if aug_type == "horizontal_flip":
            img = cv2.flip(img, 1)
            boxes = [[cls_id, 1.0 - cx, cy, bw, bh] for cls_id, cx, cy, bw, bh in boxes]

        elif aug_type == "vertical_flip":
            img = cv2.flip(img, 0)
            boxes = [[cls_id, cx, 1.0 - cy, bw, bh] for cls_id, cx, cy, bw, bh in boxes]

        elif aug_type == "rotate_90_cw":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # Transform: (cx, cy) -> (cy, 1-cx), swap w and h
            boxes = [[cls_id, cy, 1.0 - cx, bh, bw] for cls_id, cx, cy, bw, bh in boxes]

        elif aug_type == "rotate_90_ccw":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Transform: (cx, cy) -> (1-cy, cx), swap w and h
            boxes = [[cls_id, 1.0 - cy, cx, bh, bw] for cls_id, cx, cy, bw, bh in boxes]

        elif aug_type == "rotate_180":
            img = cv2.rotate(img, cv2.ROTATE_180)
            # Transform: (cx, cy) -> (1-cx, 1-cy)
            boxes = [[cls_id, 1.0 - cx, 1.0 - cy, bw, bh] for cls_id, cx, cy, bw, bh in boxes]

        # Convert boxes back to label format
        aug_labels = []
        for cls_id, cx, cy, bw, bh in boxes:
            # Clip coordinates to valid range
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            bw = np.clip(bw, 0.0, 1.0)
            bh = np.clip(bh, 0.0, 1.0)

            aug_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        return img, aug_labels

    def _generate_oversampling_report(
        self, original_dir, oversampled_dir, replication_plan, total_augmented
    ):
        """
        Generate comprehensive oversampling report.

        Args:
            original_dir (str): Original dataset directory
            oversampled_dir (str): Oversampled dataset directory
            replication_plan (dict): Replication plan details
            total_augmented (int): Total number of augmented images
        """
        report_path = os.path.join(self.save_dir, "oversampling_report.txt")

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("OVERSAMPLING REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Original dataset: {original_dir}\n")
            f.write(f"Oversampled dataset: {oversampled_dir}\n")
            f.write(f"Total augmented images: {total_augmented}\n\n")

            f.write("Replication plan:\n")
            f.write("-" * 70 + "\n")
            f.write(
                f"{'Class':<20} {'Original':<12} {'Target':<12} {'Factor':<10} {'Images':<10}\n"
            )
            f.write("-" * 70 + "\n")

            for class_name, plan in replication_plan.items():
                f.write(
                    f"{class_name:<20} {plan['current']:<12} {plan['target']:<12} "
                    f"{plan['factor']:<10} {len(plan['images']):<10}\n"
                )

            f.write("\n" + "=" * 70 + "\n")
            f.write("Augmentation types applied:\n")
            f.write("  - Horizontal flip\n")
            f.write("  - Vertical flip\n")
            f.write("  - 90 degree clockwise rotation\n")
            f.write("  - 90 degree counter-clockwise rotation\n")
            f.write("  - 180 degree rotation (upside down)\n")
            f.write("=" * 70 + "\n")

        print(f"\nOversampling report saved: {report_path}")

    def validate_detection_model(self, config_path):
        """Validate the trained detection model."""
        if self.model is None:
            print("No model to validate")
            return None

        print("\nValidating detection model...")
        try:
            # Run validation
            results = self.model.val(data=config_path, conf=self.default_conf)

            print("Validation completed successfully!")
            print(f"mAP50: {results.box.map50:.3f}")
            print(f"mAP50-95: {results.box.map:.3f}")

            # Class-wise metrics
            if hasattr(results.box, "ap_class_index"):
                print("\nClass-wise mAP50:")
                for i, class_idx in enumerate(results.box.ap_class_index):
                    if class_idx < len(self.class_names):
                        class_name = self.class_names[class_idx]
                        ap = results.box.ap50[i] if i < len(results.box.ap50) else 0
                        print(f"  {class_name}: {ap:.3f}")

            return results

        except Exception as e:
            print(f"Validation failed: {e}")
            return None

    def predict_with_detection_model(self, source_path, save_results=True):
        """Run predictions with the detection model."""
        if self.model is None:
            print("No model available for prediction")
            return None

        print(f"\nRunning detection predictions on: {source_path}")

        try:
            results = self.model.predict(
                source=source_path,
                conf=self.default_conf,
                iou=0.5,
                max_det=350,
                save=save_results,
                project=self.save_dir,
                name="predictions",
            )

            print(f"Predictions completed! Results saved in: {self.save_dir}/predictions")
            return results

        except Exception as e:
            print(f"Prediction failed: {e}")
            return None


# Main training function for detection
def main_detection_training():
    """
    Main detection training pipeline with aggressive class weighting, progressive unfreezing,
    and minority class oversampling.
    """
    try:
        print("Initializing YOLOv12 Detection Trainer...")
        trainer = AdaptiveYOLOv12DetectionTrainer(model_size="s", img_size=640, batch_size=10)

        print("\nStep 1: Downloading dataset...")
        gdrive_file_id = "11BZGKQFbwo5wT9d1zlWbYqzSV8MoMP2B"
        dataset_path = trainer.download_and_extract_dataset(gdrive_file_id)

        print("\nStep 2: Preparing detection dataset...")
        config_path = trainer.prepare_detection_dataset(dataset_path, min_area=0.0, val_split=0.2)

        # Extract the dataset directory from config path
        prepared_dataset_dir = os.path.dirname(config_path)

        print("\nStep 3: Applying aggressive minority oversampling...")
        oversampled_dataset_dir = trainer.aggressive_minority_oversampling(
            dataset_dir=prepared_dataset_dir, target_samples_per_class=5000, minority_threshold=2500
        )

        # Update config path to point to oversampled dataset
        config_path = os.path.join(oversampled_dataset_dir, "data.yaml")
        print(f"Using oversampled dataset config: {config_path}")

        print("\nStep 4: Initializing YOLO detection model...")
        if not trainer.initialize_yolo_detection_model():
            raise RuntimeError("Failed to initialize detection model")

        print("\nStep 5: Training with progressive unfreezing...")
        custom_schedule = {
            0: 0.20,  # Detection heads only (0-30 epochs)
            30: 0.40,  # + Neck PAN (30-60 epochs)
            60: 0.60,  # + Neck FPN (60-100 epochs)
            100: 0.75,  # + Late backbone (100-140 epochs)
            140: 0.90,  # + Mid backbone (140-180 epochs)
            180: 1.0,  # Full model (180-210 epochs)
        }

        training_results = trainer.train_detection_model_with_progressive_unfreezing(
            config_path, epochs=210, unfreeze_schedule=custom_schedule
        )

        print("\nStep 6: Comprehensive validation of all checkpoints...")
        validation_results = trainer.validate_all_checkpoints(config_path)

        if validation_results:
            print(f"\nBest model available at: {validation_results['best_model_path']}")
            print(f"   Use this model for inference!")

        # Cleanup
        if os.path.exists("dataset/"):
            shutil.rmtree("dataset/")

        print("\n" + "=" * 60)
        print("DETECTION TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Features implemented:")
        print("  - Pure YOLO detection (no classification confusion)")
        print("  - Aggressive class weighting for imbalanced datasets")
        print("  - Minority class oversampling with strong augmentations")
        print("  - Detection-optimized progressive unfreezing")
        print("  - Proper YOLO architecture understanding")
        print("=" * 60)

        return trainer

    except Exception as e:
        print(f"\nERROR: Detection training failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    trained_model = main_detection_training()
