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


class EnhancedWeightedYOLOv12Classifier:
    """
    Enhanced YOLOv12 classifier with optimized confidence threshold management
    and overlapping bounding box handling capabilities.
    """

    def __init__(self, model_size="n", img_size=640, batch_size=12, default_conf=0.25):
        """
        Initialize the Enhanced WeightedYOLOv12Classifier.

        Args:
            model_size (str): YOLOv12 model size ('n', 's', 'm', 'l', 'x')
            img_size (int): Image size for training
            batch_size (int): Batch size for training
            default_conf (float): Default confidence threshold
        """
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.default_conf = default_conf

        # Confidence threshold management
        self.confidence_thresholds = {
            "training": 0.25,
            "validation": 0.4,
            "inference": 0.5,
            "strict": 0.7,
        }

        self.model = None
        self.class_names = []
        self.original_class_names = []
        self.class_weights = None
        self.class_weights_tensor = None
        self.weighted_loss_fn = None
        self.id_map = {}

        # Create folder with timestamp for this training session
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/EnhancedYOLOv12_Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def set_confidence_threshold(self, threshold_type="inference", custom_value=None):
        """
        Set confidence threshold efficiently for different use cases.

        Args:
            threshold_type (str): Type of threshold ('training', 'validation', 'inference', 'strict')
            custom_value (float): Custom threshold value (overrides threshold_type)

        Returns:
            float: The set confidence threshold
        """
        if custom_value is not None:
            if not 0.0 <= custom_value <= 1.0:
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            self.default_conf = custom_value
            print(f"Custom confidence threshold set to: {custom_value}")
        else:
            if threshold_type not in self.confidence_thresholds:
                raise ValueError(
                    f"Invalid threshold type. Choose from: {list(self.confidence_thresholds.keys())}"
                )
            self.default_conf = self.confidence_thresholds[threshold_type]
            print(f"Confidence threshold set to {threshold_type}: {self.default_conf}")

        return self.default_conf

    def optimize_confidence_threshold(self, validation_path, threshold_range=(0.1, 0.9), step=0.05):
        """
        Automatically find optimal confidence threshold based on validation data.

        Args:
            validation_path (str): Path to validation dataset
            threshold_range (tuple): Range of thresholds to test
            step (float): Step size for threshold testing

        Returns:
            dict: Optimization results with best threshold and metrics
        """
        if self.model is None:
            print("Error: Model not initialized for threshold optimization")
            return None

        print("Optimizing confidence threshold...")

        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        results = []

        # Get validation images
        val_images_path = os.path.join(validation_path, "val", "images")
        if not os.path.exists(val_images_path):
            print(f"Validation images not found at: {val_images_path}")
            return None

        val_images = [
            f for f in os.listdir(val_images_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][
            :100
        ]  # Sample for efficiency

        print(f"Testing {len(thresholds)} thresholds on {len(val_images)} validation images...")

        for threshold in thresholds:
            correct_predictions = 0
            total_predictions = 0
            confident_predictions = 0

            for img_file in val_images:
                img_path = os.path.join(val_images_path, img_file)

                # Determine true class from filename
                true_class_idx = None
                for class_idx, class_name in enumerate(self.class_names):
                    if class_name.lower() in img_file.lower():
                        true_class_idx = class_idx
                        break

                if true_class_idx is None:
                    continue

                try:
                    # Run prediction with current threshold
                    results_pred = self.model.predict(img_path, verbose=False, conf=threshold)

                    if results_pred and len(results_pred) > 0:
                        probs = results_pred[0].probs
                        if probs is not None:
                            confident_predictions += 1
                            predicted_class = probs.top1
                            if predicted_class == true_class_idx:
                                correct_predictions += 1

                    total_predictions += 1

                except Exception as e:
                    continue

            # Calculate metrics
            accuracy = correct_predictions / max(total_predictions, 1)
            confidence_rate = confident_predictions / max(total_predictions, 1)

            # Combined score balancing accuracy and confidence
            combined_score = accuracy * 0.7 + confidence_rate * 0.3

            results.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy,
                    "confidence_rate": confidence_rate,
                    "combined_score": combined_score,
                    "correct_predictions": correct_predictions,
                    "total_predictions": total_predictions,
                }
            )

            print(f"Threshold {threshold:.2f}: Acc={accuracy:.3f}, Conf_rate={confidence_rate:.3f}")

        # Find best threshold
        best_result = max(results, key=lambda x: x["combined_score"])
        optimal_threshold = best_result["threshold"]

        # Update default confidence
        self.default_conf = optimal_threshold

        # Save optimization results
        optimization_results = {
            "optimal_threshold": optimal_threshold,
            "best_metrics": best_result,
            "all_results": results,
            "optimization_date": datetime.datetime.now().isoformat(),
        }

        results_path = os.path.join(self.save_dir, "threshold_optimization.json")
        with open(results_path, "w") as f:
            json.dump(optimization_results, f, indent=2)

        # Create visualization
        self._plot_threshold_optimization(results)

        print(f"Optimal confidence threshold found: {optimal_threshold}")
        print(f"Best combined score: {best_result['combined_score']:.3f}")
        print(f"Optimization results saved to: {results_path}")

        return optimization_results

    def _plot_threshold_optimization(self, results):
        """
        Plot threshold optimization results.

        Args:
            results (list): List of threshold optimization results
        """
        thresholds = [r["threshold"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        confidence_rates = [r["confidence_rate"] for r in results]
        combined_scores = [r["combined_score"] for r in results]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(thresholds, accuracies, "b-o", label="Accuracy")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Confidence Threshold")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(thresholds, confidence_rates, "g-s", label="Confidence Rate")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Confidence Rate")
        plt.title("Confidence Rate vs Threshold")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(thresholds, combined_scores, "r-^", label="Combined Score")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Combined Score")
        plt.title("Combined Score vs Threshold")
        plt.grid(True, alpha=0.3)

        # Find and mark optimal threshold
        best_idx = np.argmax(combined_scores)
        optimal_threshold = thresholds[best_idx]
        plt.axvline(
            x=optimal_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal: {optimal_threshold:.2f}",
        )
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(thresholds, accuracies, "b-o", label="Accuracy", alpha=0.7)
        plt.plot(thresholds, confidence_rates, "g-s", label="Confidence Rate", alpha=0.7)
        plt.plot(thresholds, combined_scores, "r-^", label="Combined Score", alpha=0.7)
        plt.axvline(
            x=optimal_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal: {optimal_threshold:.2f}",
        )
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric Value")
        plt.title("All Metrics Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "threshold_optimization_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Threshold optimization plot saved to: {plot_path}")

    def handle_overlapping_bboxes(self, bboxes_yolo, overlap_threshold=0.5, strategy="nms"):
        """
        Handle overlapping bounding boxes using various strategies.

        Args:
            bboxes_yolo (list): List of YOLO format bboxes [class_id, cx, cy, w, h]
            overlap_threshold (float): IoU threshold for considering boxes as overlapping
            strategy (str): Strategy to handle overlaps ('nms', 'merge', 'keep_largest', 'keep_highest_conf')

        Returns:
            list: Processed bounding boxes
        """
        if len(bboxes_yolo) <= 1:
            return bboxes_yolo

        # Convert YOLO format to [x1, y1, x2, y2] for easier processing
        boxes_xyxy = []
        for bbox in bboxes_yolo:
            if len(bbox) >= 5:
                class_id, cx, cy, w, h = bbox[:5]
                conf = bbox[5] if len(bbox) > 5 else 1.0

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                boxes_xyxy.append([x1, y1, x2, y2, class_id, conf])
            else:
                print(f"Warning: Invalid bbox format: {bbox}")
                continue

        if len(boxes_xyxy) <= 1:
            return bboxes_yolo

        processed_boxes = []

        if strategy == "nms":
            processed_boxes = self._apply_nms(boxes_xyxy, overlap_threshold)
        elif strategy == "merge":
            processed_boxes = self._merge_overlapping_boxes(boxes_xyxy, overlap_threshold)
        elif strategy == "keep_largest":
            processed_boxes = self._keep_largest_boxes(boxes_xyxy, overlap_threshold)
        elif strategy == "keep_highest_conf":
            processed_boxes = self._keep_highest_confidence_boxes(boxes_xyxy, overlap_threshold)
        else:
            print(f"Unknown strategy: {strategy}. Using NMS as default.")
            processed_boxes = self._apply_nms(boxes_xyxy, overlap_threshold)

        # Convert back to YOLO format
        result_yolo = []
        for box in processed_boxes:
            x1, y1, x2, y2, class_id, conf = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            result_yolo.append([int(class_id), cx, cy, w, h, conf])

        return result_yolo

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes.

        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2, class_id, conf]

        Returns:
            float: IoU value
        """
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0

        intersection = (x2_min - x1_max) * (y2_min - y1_max)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-8)

    def _apply_nms(self, boxes, iou_threshold):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.

        Args:
            boxes (list): List of boxes in format [x1, y1, x2, y2, class_id, conf]
            iou_threshold (float): IoU threshold

        Returns:
            list: Filtered boxes
        """
        if len(boxes) == 0:
            return []

        # Sort by confidence (descending)
        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)

        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)

            # Remove boxes with high IoU with current box and same class
            boxes = [
                box
                for box in boxes
                if box[4] != current[4]  # Different class
                or self._calculate_iou(current, box) < iou_threshold
            ]

        return keep

    def _merge_overlapping_boxes(self, boxes, iou_threshold):
        """
        Merge overlapping boxes of the same class.

        Args:
            boxes (list): List of boxes
            iou_threshold (float): IoU threshold

        Returns:
            list: Merged boxes
        """
        if len(boxes) == 0:
            return []

        merged = []
        processed = set()

        for i, box1 in enumerate(boxes):
            if i in processed:
                continue

            merge_group = [box1]
            processed.add(i)

            for j, box2 in enumerate(boxes[i + 1 :], i + 1):
                if j in processed:
                    continue

                # Only merge boxes of the same class
                if box1[4] == box2[4] and self._calculate_iou(box1, box2) >= iou_threshold:
                    merge_group.append(box2)
                    processed.add(j)

            # Merge boxes in group
            if len(merge_group) == 1:
                merged.append(merge_group[0])
            else:
                merged_box = self._merge_box_group(merge_group)
                merged.append(merged_box)

        return merged

    def _merge_box_group(self, box_group):
        """
        Merge a group of overlapping boxes into one box.

        Args:
            box_group (list): List of boxes to merge

        Returns:
            list: Single merged box
        """
        if len(box_group) == 1:
            return box_group[0]

        # Calculate weighted average coordinates based on confidence
        total_conf = sum(box[5] for box in box_group)

        x1 = sum(box[0] * box[5] for box in box_group) / total_conf
        y1 = sum(box[1] * box[5] for box in box_group) / total_conf
        x2 = sum(box[2] * box[5] for box in box_group) / total_conf
        y2 = sum(box[3] * box[5] for box in box_group) / total_conf

        # Use the class and max confidence
        class_id = box_group[0][4]  # Assuming same class
        max_conf = max(box[5] for box in box_group)

        return [x1, y1, x2, y2, class_id, max_conf]

    def _keep_largest_boxes(self, boxes, iou_threshold):
        """
        Keep the largest box among overlapping boxes of the same class.

        Args:
            boxes (list): List of boxes
            iou_threshold (float): IoU threshold

        Returns:
            list: Filtered boxes
        """
        if len(boxes) == 0:
            return []

        # Calculate area for each box
        boxes_with_area = []
        for box in boxes:
            area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_with_area.append(box + [area])

        # Sort by area (descending)
        boxes_with_area = sorted(boxes_with_area, key=lambda x: x[6], reverse=True)

        keep = []
        processed = set()

        for i, box1 in enumerate(boxes_with_area):
            if i in processed:
                continue

            keep.append(box1[:6])  # Remove area from result
            processed.add(i)

            # Mark overlapping boxes as processed
            for j, box2 in enumerate(boxes_with_area[i + 1 :], i + 1):
                if j in processed:
                    continue

                if (
                    box1[4] == box2[4]  # Same class
                    and self._calculate_iou(box1[:6], box2[:6]) >= iou_threshold
                ):
                    processed.add(j)

        return keep

    def _keep_highest_confidence_boxes(self, boxes, iou_threshold):
        """
        Keep the highest confidence box among overlapping boxes.

        Args:
            boxes (list): List of boxes
            iou_threshold (float): IoU threshold

        Returns:
            list: Filtered boxes
        """
        # This is essentially NMS, so we can reuse it
        return self._apply_nms(boxes, iou_threshold)

    def clean_and_extract_objects_with_overlap_handling(
        self,
        dataset_path,
        min_area=0.0001,
        max_samples_per_class=15000,
        val_split=0.2,
        overlap_strategy="nms",
        overlap_threshold=0.5,
    ):
        """
        Enhanced version of clean_and_extract_objects with overlap handling.

        Args:
            dataset_path (str): Path to original YOLO dataset
            min_area (float): Minimum bbox area threshold
            max_samples_per_class (int): Maximum samples per class
            val_split (float): Validation split ratio
            overlap_strategy (str): Strategy for handling overlapping boxes
            overlap_threshold (float): IoU threshold for overlap detection

        Returns:
            str: Path to cleaned classification dataset
        """
        print("Extracting and cleaning detection annotations with overlap handling...")

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
        background_keywords = ["background", "bg", "__background__", "void", "unlabeled"]

        # Filter out background classes
        filtered_classes = []
        self.id_map = {}
        new_class_id = 0

        for original_id, class_name in enumerate(original_names):
            if class_name not in background_keywords:
                filtered_classes.append(class_name)
                self.id_map[original_id] = new_class_id
                new_class_id += 1
            else:
                print(f"Removing background class: '{class_name}' (original ID: {original_id})")

        self.class_names = filtered_classes

        if len(self.class_names) == 0:
            raise ValueError("No valid classes found after background removal!")

        print(f"Final target classes: {self.class_names}")
        print(f"Overlap handling strategy: {overlap_strategy} (threshold: {overlap_threshold})")

        # Prepare cleaned dataset directory
        classification_dir = os.path.join("cleaned_dataset_with_overlap_handling")

        if os.path.exists(classification_dir):
            shutil.rmtree(classification_dir)

        os.makedirs(classification_dir, exist_ok=True)

        # Initialize tracking variables
        class_counts = {name: 0 for name in self.class_names}
        overlap_stats = {
            "images_with_overlaps": 0,
            "total_overlaps_detected": 0,
            "overlaps_resolved": 0,
        }

        # Process train split
        train_images_dir = os.path.join(dataset_path, "train", "images")
        train_labels_dir = os.path.join(dataset_path, "train", "labels")

        if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
            raise FileNotFoundError("Train images or labels directory not found")

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

        train_imgs, val_imgs = self._create_stratified_split(
            all_images, train_labels_dir, self.id_map, val_split
        )

        print(f"Split created: {len(train_imgs)} train, {len(val_imgs)} val")

        # Process training images with overlap handling
        train_stats = self._process_image_split_with_overlap_handling(
            train_imgs,
            train_images_dir,
            train_labels_dir,
            new_train_images_dir,
            new_train_labels_dir,
            self.id_map,
            min_area,
            overlap_strategy,
            overlap_threshold,
        )

        # Process validation images with overlap handling
        val_stats = self._process_image_split_with_overlap_handling(
            val_imgs,
            train_images_dir,
            train_labels_dir,
            new_val_images_dir,
            new_val_labels_dir,
            self.id_map,
            min_area,
            overlap_strategy,
            overlap_threshold,
        )

        # Combine statistics
        total_labels = train_stats["total"] + val_stats["total"]
        kept_labels = train_stats["kept"] + val_stats["kept"]

        for key in overlap_stats:
            overlap_stats[key] = train_stats[key] + val_stats[key]

        for class_name in self.class_names:
            class_counts[class_name] = (
                train_stats["class_counts"][class_name] + val_stats["class_counts"][class_name]
            )

        # Calculate class weights
        self._calculate_class_weights(class_counts)

        print(f"\nObject cleaning summary with overlap handling:")
        print(f"Original labels: {total_labels}")
        print(f"Kept labels: {kept_labels}")
        print(f"Images with overlaps: {overlap_stats['images_with_overlaps']}")
        print(f"Total overlaps detected: {overlap_stats['total_overlaps_detected']}")
        print(f"Overlaps resolved: {overlap_stats['overlaps_resolved']}")

        for cls, count in class_counts.items():
            print(f"  {cls}: {count} objects")

        return classification_dir

    def _process_image_split_with_overlap_handling(
        self,
        images,
        src_img_dir,
        src_labels_dir,
        dst_img_dir,
        dst_labels_dir,
        id_map,
        min_area,
        overlap_strategy,
        overlap_threshold,
    ):
        """
        Process image split with overlap handling capabilities.

        Args:
            images (list): List of image filenames to process
            src_img_dir (str): Source images directory
            src_labels_dir (str): Source labels directory
            dst_img_dir (str): Destination images directory
            dst_labels_dir (str): Destination labels directory
            id_map (dict): Class ID mapping
            min_area (float): Minimum bbox area threshold
            overlap_strategy (str): Strategy for handling overlaps
            overlap_threshold (float): IoU threshold for overlap detection

        Returns:
            dict: Processing statistics including overlap information
        """
        class_counts = {name: 0 for name in self.class_names}
        total_labels, kept_labels = 0, 0
        images_with_valid_labels = 0

        overlap_stats = {
            "images_with_overlaps": 0,
            "total_overlaps_detected": 0,
            "overlaps_resolved": 0,
        }

        for img_file in images:
            img_path = os.path.join(src_img_dir, img_file)
            new_img_path = os.path.join(dst_img_dir, img_file)

            label_file = os.path.splitext(img_file)[0] + ".txt"
            old_label_path = os.path.join(src_labels_dir, label_file)
            new_label_path = os.path.join(dst_labels_dir, label_file)

            if os.path.exists(old_label_path):
                processed_labels = self._process_label_file_with_overlap_handling(
                    old_label_path, id_map, min_area, overlap_strategy, overlap_threshold
                )

                # Update overlap statistics
                for key in overlap_stats:
                    overlap_stats[key] += processed_labels.get(key, 0)

                if processed_labels["lines"]:
                    shutil.copy2(img_path, new_img_path)
                    images_with_valid_labels += 1

                    with open(new_label_path, "w") as f:
                        f.writelines(processed_labels["lines"])

                    total_labels += processed_labels["total"]
                    kept_labels += processed_labels["kept"]

                    for class_name, count in processed_labels["class_counts"].items():
                        class_counts[class_name] += count

        print(
            f"  Processed {images_with_valid_labels} images with valid labels out of {len(images)} total images"
        )

        result_stats = {"total": total_labels, "kept": kept_labels, "class_counts": class_counts}
        result_stats.update(overlap_stats)

        return result_stats

    def _process_label_file_with_overlap_handling(
        self, label_path, id_map, min_area, overlap_strategy, overlap_threshold
    ):
        """
        Process label file with overlap handling.

        Args:
            label_path (str): Path to label file
            id_map (dict): Class ID mapping
            min_area (float): Minimum bbox area threshold
            overlap_strategy (str): Strategy for handling overlaps
            overlap_threshold (float): IoU threshold

        Returns:
            dict: Processing results including overlap statistics
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
                "images_with_overlaps": 0,
                "total_overlaps_detected": 0,
                "overlaps_resolved": 0,
            }

        # Parse all bounding boxes
        bboxes = []
        total_lines = len(lines)

        for line_num, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            # Skip classes not in our mapping or invalid boxes
            if class_id not in id_map:
                continue

            if bw <= 0 or bh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1 or bw * bh < min_area:
                continue

            # Add confidence score (default to 1.0 for ground truth)
            bboxes.append([class_id, cx, cy, bw, bh, 1.0])

        overlap_stats = {
            "images_with_overlaps": 0,
            "total_overlaps_detected": 0,
            "overlaps_resolved": 0,
        }

        # Handle overlapping boxes if there are multiple boxes
        if len(bboxes) > 1:
            # Detect overlaps
            overlaps_detected = 0
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    if self._calculate_yolo_iou(bboxes[i], bboxes[j]) >= overlap_threshold:
                        overlaps_detected += 1

            if overlaps_detected > 0:
                overlap_stats["images_with_overlaps"] = 1
                overlap_stats["total_overlaps_detected"] = overlaps_detected

                # Apply overlap handling
                original_count = len(bboxes)
                bboxes = self.handle_overlapping_bboxes(bboxes, overlap_threshold, overlap_strategy)
                overlap_stats["overlaps_resolved"] = original_count - len(bboxes)

        # Convert back to lines and update class counts
        new_lines = []
        class_counts = {name: 0 for name in self.class_names}

        for bbox in bboxes:
            class_id, cx, cy, bw, bh = bbox[:5]
            new_id = id_map[class_id]
            new_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            if new_id < len(self.class_names):
                class_counts[self.class_names[new_id]] += 1

        result = {
            "lines": new_lines,
            "total": total_lines,
            "kept": len(new_lines),
            "class_counts": class_counts,
        }
        result.update(overlap_stats)

        return result

    def _calculate_yolo_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two YOLO format bounding boxes.

        Args:
            bbox1, bbox2: YOLO format boxes [class_id, cx, cy, w, h, conf]

        Returns:
            float: IoU value
        """
        # Convert to x1, y1, x2, y2 format
        cx1, cy1, w1, h1 = bbox1[1:5]
        cx2, cy2, w2, h2 = bbox2[1:5]

        x1_1, y1_1 = cx1 - w1 / 2, cy1 - h1 / 2
        x2_1, y2_1 = cx1 + w1 / 2, cy1 + h1 / 2

        x1_2, y1_2 = cx2 - w2 / 2, cy2 - h2 / 2
        x2_2, y2_2 = cx2 + w2 / 2, cy2 + h2 / 2

        # Calculate intersection
        x1_max = max(x1_1, x1_2)
        y1_max = max(y1_1, y1_2)
        x2_min = min(x2_1, x2_2)
        y2_min = min(y2_1, y2_2)

        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0

        intersection = (x2_min - x1_max) * (y2_min - y1_max)

        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-8)

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

                    img_class = None
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id in id_map:
                                img_class = id_map[class_id]
                                break

                    if img_class is not None:
                        valid_images.append(img_file)
                        image_classes.append(img_class)

            print(f"Found {len(valid_images)} images with valid classes for stratification")

            if len(valid_images) > 0 and len(set(image_classes)) > 1:
                train_imgs, val_imgs = train_test_split(
                    valid_images, test_size=val_split, stratify=image_classes, random_state=42
                )
                return train_imgs, val_imgs
            else:
                raise ValueError("Insufficient data for stratified split")

        except Exception as e:
            print(f"Stratified split failed: {e}. Using random split...")
            train_imgs, val_imgs = train_test_split(
                all_images, test_size=val_split, random_state=42
            )
            return train_imgs, val_imgs

    def _calculate_class_weights(self, class_counts):
        """
        Calculate class weights to handle imbalanced datasets.

        Args:
            class_counts (dict): Dictionary with class counts
        """
        if not class_counts or sum(class_counts.values()) == 0:
            print("Warning: No class counts available for weight calculation")
            return

        empty_classes = [cls for cls, count in class_counts.items() if count == 0]
        if empty_classes:
            print(f"Warning: Classes with no samples (will get default weight): {empty_classes}")

        valid_classes = {cls: count for cls, count in class_counts.items() if count > 0}

        if len(valid_classes) == 0:
            print("Error: No classes have any samples!")
            return

        counts = np.array([valid_classes[name] for name in valid_classes.keys()])
        class_labels = np.arange(len(valid_classes))

        weights = compute_class_weight(
            class_weight="balanced", classes=class_labels, y=np.repeat(class_labels, counts)
        )

        self.class_weights = {}
        default_weight = np.mean(weights) if len(weights) > 0 else 1.0
        valid_class_names = list(valid_classes.keys())

        for i, class_name in enumerate(self.class_names):
            if class_name in valid_class_names:
                weight_idx = valid_class_names.index(class_name)
                self.class_weights[i] = float(weights[weight_idx])
            else:
                self.class_weights[i] = default_weight

        weight_values = [self.class_weights[i] for i in range(len(self.class_names))]
        self.class_weights_tensor = torch.FloatTensor(weight_values)

        print("Class weights calculated:")
        for i, (class_name, weight) in enumerate(zip(self.class_names, weight_values)):
            sample_count = class_counts.get(class_name, 0)
            print(f"  {class_name}: {weight:.3f} (samples: {sample_count})")

    def create_yolo_classification_config(self, classification_dir):
        """
        Create YOLO classification configuration file.

        Args:
            classification_dir (str): Path to classification dataset

        Returns:
            str: Path to configuration file
        """
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

    def initialize_yolov12_classifier(self):
        """
        Initialize YOLOv12 model for classification with class weights.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if len(self.class_names) == 0:
            print("Error: No classes defined. Cannot initialize model.")
            return False

        try:
            model_name = f"yolo12{self.model_size}-cls.pt"
            print(f"Attempting to load {model_name}...")
            self.model = YOLO(model_name)
            print(f"YOLOv12{self.model_size} Classification model initialized successfully")

            if self.class_weights_tensor is not None:
                self.patch_model_loss()

            return True

        except Exception as e:
            print(f"Error initializing YOLOv12 classifier: {e}")
            try:
                model_name = f"yolo12{self.model_size}.pt"
                print(f"Trying fallback detection model: {model_name}")
                self.model = YOLO(model_name)
                print(f"Using YOLOv12{self.model_size} detection model as fallback")

                if self.class_weights_tensor is not None:
                    self.patch_model_loss()

                return True

            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                return False

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
            device = next(self.model.model.parameters()).device
            weighted_tensor = self.class_weights_tensor.to(device)

            print(f"Applying class weights on device: {device}")

            if hasattr(self.model.model, "loss"):
                original_loss = self.model.model.loss

                def weighted_loss_wrapper(*args, **kwargs):
                    loss_dict = original_loss(*args, **kwargs)

                    if "cls" in loss_dict and len(args) >= 2:
                        preds, targets = args[0], args[1]
                        if hasattr(targets, "cls"):
                            cls_targets = targets.cls.long()
                            cls_preds = preds[0] if isinstance(preds, (list, tuple)) else preds

                            if cls_preds.size(1) != len(self.class_names):
                                return loss_dict

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

    def apply_progressive_unfreezing(self, unfreeze_schedule=None):
        """
        Apply progressive unfreezing strategy to the YOLOv12 model.

        Args:
            unfreeze_schedule (dict): Schedule for unfreezing layers

        Returns:
            dict: Unfreezing configuration
        """
        if self.model is None:
            print("Error: Model not initialized. Call initialize_yolov12_classifier() first.")
            return None

        if unfreeze_schedule is None:
            unfreeze_schedule = {
                0: 0.1,
                20: 0.3,
                50: 0.6,
                75: 1.0,
            }

        print("Setting up progressive unfreezing strategy:")
        for epoch, percentage in sorted(unfreeze_schedule.items()):
            print(f"  Epoch {epoch}: {percentage*100:.0f}% of layers will be unfrozen")

        total_params = []
        param_names = []

        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                total_params.append(param)
                param_names.append(name)

        total_layer_count = len(total_params)
        print(f"Total trainable parameters found: {total_layer_count}")

        initial_unfreeze_ratio = unfreeze_schedule.get(0, 0.1)
        layers_to_unfreeze = int(total_layer_count * initial_unfreeze_ratio)

        for param in total_params:
            param.requires_grad = False

        for i in range(total_layer_count - layers_to_unfreeze, total_layer_count):
            total_params[i].requires_grad = True

        frozen_count = sum(1 for param in total_params if not param.requires_grad)
        unfrozen_count = total_layer_count - frozen_count

        print(f"Initial state: {unfrozen_count} layers unfrozen, {frozen_count} layers frozen")

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

        Args:
            current_epoch (int): Current training epoch

        Returns:
            bool: True if unfreezing status was updated, False otherwise
        """
        if not hasattr(self, "unfreeze_schedule") or not hasattr(self, "total_params"):
            print("Warning: Progressive unfreezing not initialized.")
            return False

        if current_epoch in self.unfreeze_schedule:
            target_ratio = self.unfreeze_schedule[current_epoch]
            target_unfrozen_count = int(self.total_layer_count * target_ratio)

            for param in self.total_params:
                param.requires_grad = False

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
        self,
        classification_path,
        epochs=100,
        unfreeze_schedule=None,
        auto_optimize_confidence=True,
        overlap_handling_enabled=True,
    ):
        """
        Enhanced unified training method with all optimization features.

        Args:
            classification_path (str): Path to classification dataset
            epochs (int): Number of training epochs
            unfreeze_schedule (dict): Custom unfreezing schedule
            auto_optimize_confidence (bool): Whether to optimize confidence threshold automatically
            overlap_handling_enabled (bool): Whether overlap handling was used during preprocessing

        Returns:
            object: Training results
        """
        print(f"\n{'='*60}")
        print("ENHANCED UNIFIED TRAINING: PROGRESSIVE UNFREEZING + CLASS WEIGHTS + OPTIMIZATIONS")
        print(f"{'='*60}")

        if len(self.class_names) == 0:
            raise ValueError("No classes defined. Cannot proceed with training.")

        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_yolov12_classifier() first.")

        # Apply progressive unfreezing strategy
        unfreezing_config = self.apply_progressive_unfreezing(unfreeze_schedule)
        if unfreezing_config is None:
            print("Failed to set up progressive unfreezing.")
            return None

        print(f"Enhanced Features Enabled:")
        print(f"  ✓ Progressive unfreezing")
        print(f"  ✓ Class weights")
        print(f"  ✓ Confidence threshold: {self.default_conf}")
        print(f"  ✓ Auto confidence optimization: {auto_optimize_confidence}")
        print(f"  ✓ Overlap handling: {overlap_handling_enabled}")

        # Create classification config
        config_path = self.create_yolo_classification_config(classification_path)

        # Base training arguments with optimized confidence
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
            "conf": self.default_conf,  # Apply optimized confidence threshold
        }

        # Phase-based training
        epoch_phases = sorted(self.unfreeze_schedule.keys())
        all_phase_results = []

        print("Starting enhanced unified training...")

        for i, phase_start_epoch in enumerate(epoch_phases):
            if i + 1 < len(epoch_phases):
                phase_epochs = epoch_phases[i + 1] - phase_start_epoch
            else:
                phase_epochs = epochs - phase_start_epoch

            if phase_epochs <= 0:
                continue

            print(f"\n{'='*50}")
            print(
                f"ENHANCED PHASE {i+1}: EPOCHS {phase_start_epoch} to {phase_start_epoch + phase_epochs - 1}"
            )
            print(f"{'='*50}")

            self.update_unfreezing_schedule(phase_start_epoch)

            if i > 0:
                previous_phase_name = f"enhanced_phase_{i}"
                last_checkpoint = os.path.join(
                    self.save_dir, previous_phase_name, "weights", "last.pt"
                )
                if os.path.exists(last_checkpoint):
                    print(f"Resuming from checkpoint: {last_checkpoint}")
                    self.model = YOLO(last_checkpoint)
                    self.patch_model_loss()
                    self.update_unfreezing_schedule(phase_start_epoch)

            phase_training_args = base_training_args.copy()
            phase_training_args.update(
                {
                    "epochs": phase_epochs,
                    "name": f"enhanced_phase_{i+1}",
                }
            )

            print(f"Starting enhanced training for phase {i+1}...")
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
                print(f"Enhanced phase {i+1} completed successfully!")

            except Exception as e:
                print(f"Error in enhanced phase {i+1}: {e}")
                continue

        # Auto-optimize confidence threshold after training
        if auto_optimize_confidence:
            print(f"\n{'='*50}")
            print("POST-TRAINING CONFIDENCE THRESHOLD OPTIMIZATION")
            print(f"{'='*50}")
            optimization_results = self.optimize_confidence_threshold(classification_path)
            if optimization_results:
                print(
                    f"Optimized confidence threshold: {optimization_results['optimal_threshold']}"
                )

        print(f"\n{'='*60}")
        print("ENHANCED UNIFIED TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")

        return {
            "all_phases": all_phase_results,
            "final_results": all_phase_results[-1]["results"] if all_phase_results else None,
            "enhanced_features_applied": True,
            "final_confidence_threshold": self.default_conf,
            "class_weights": self.class_weights,
            "final_classes": self.class_names,
            "class_mapping": self.id_map,
        }

    def predict_with_confidence_management(
        self, image_path, confidence_type="inference", custom_conf=None
    ):
        """
        Run prediction with managed confidence thresholds.

        Args:
            image_path (str): Path to image
            confidence_type (str): Type of confidence to use
            custom_conf (float): Custom confidence value

        Returns:
            dict: Prediction results with confidence information
        """
        if self.model is None:
            print("Error: Model not initialized")
            return None

        # Set appropriate confidence
        if custom_conf is not None:
            conf = custom_conf
        else:
            conf = self.confidence_thresholds.get(confidence_type, self.default_conf)

        print(f"Running prediction with {confidence_type} confidence: {conf}")

        try:
            results = self.model.predict(image_path, verbose=False, conf=conf)

            if results and len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    top_indices = probs.top5
                    top_confidences = probs.top5conf.cpu().numpy()

                    predictions = []
                    for idx, confidence in zip(top_indices, top_confidences):
                        if idx < len(self.class_names):
                            predictions.append(
                                {
                                    "class_name": self.class_names[idx],
                                    "class_id": int(idx),
                                    "confidence": float(confidence),
                                }
                            )

                    return {
                        "predictions": predictions,
                        "confidence_threshold_used": conf,
                        "confidence_type": confidence_type,
                        "top_prediction": predictions[0] if predictions else None,
                    }

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

        return {
            "predictions": [],
            "confidence_threshold_used": conf,
            "confidence_type": confidence_type,
        }

    def save_enhanced_model_info(self):
        """
        Save comprehensive enhanced model information and metadata.
        """
        possible_best_paths = [
            os.path.join(self.save_dir, "enhanced_phase_1", "weights", "best.pt"),
        ]

        best_model_path = None
        for path in possible_best_paths:
            if os.path.exists(path):
                best_model_path = path
                break

        metadata = {
            "model_type": f"Enhanced Weighted YOLOv12{self.model_size} Classifier",
            "original_classes": self.original_class_names,
            "final_classes": self.class_names,
            "class_mapping": self.id_map,
            "confidence_thresholds": self.confidence_thresholds,
            "current_confidence": self.default_conf,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "num_original_classes": len(self.original_class_names),
            "num_final_classes": len(self.class_names),
            "class_weights_applied": self.class_weights,
            "training_timestamp": datetime.datetime.now().isoformat(),
            "save_directory": self.save_dir,
            "model_path": best_model_path,
            "enhanced_features": [
                "Optimized confidence threshold management",
                "Automatic confidence threshold optimization",
                "Advanced overlapping bounding box handling with multiple strategies",
                "Progressive unfreezing with class weights",
                "Enhanced prediction interface with confidence types",
                "Comprehensive overlap statistics and analysis",
                "Multi-strategy overlap resolution (NMS, merge, largest, highest confidence)",
            ],
        }

        metadata_path = os.path.join(self.save_dir, "enhanced_model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Enhanced model metadata saved to: {metadata_path}")


# Enhanced main function with all optimizations
def main_enhanced():
    """
    Main training pipeline with all enhanced features and optimizations.
    """
    try:
        print("Initializing Enhanced WeightedYOLOv12Classifier...")
        classifier = EnhancedWeightedYOLOv12Classifier(model_size="n", img_size=640, batch_size=12)

        # Set initial confidence threshold
        classifier.set_confidence_threshold("training")  # Use training threshold initially

        print("\nStep 1: Downloading dataset...")
        gdrive_file_id = "1zvCNOz4P0QFdOAfpDpSpidFVjXHFxDUE"
        dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

        print("\nStep 2: Enhanced dataset cleaning with overlap handling...")
        classification_path = classifier.clean_and_extract_objects_with_overlap_handling(
            dataset_path,
            min_area=0.0001,
            max_samples_per_class=15000,
            val_split=0.2,
            overlap_strategy="nms",  # It can be changed to 'merge', 'keep_largest', 'keep_highest_conf'
            overlap_threshold=0.5,
        )

        print("\nStep 3: Initializing YOLOv12 model...")
        if not classifier.initialize_yolov12_classifier():
            raise RuntimeError("Failed to initialize YOLOv12 model")

        print("\nStep 4: Enhanced unified training with all optimizations...")
        custom_schedule = {
            0: 0.15,
            25: 0.4,
            60: 0.7,
            85: 1.0,
        }

        training_results = classifier.train_model_with_progressive_unfreezing_and_class_weights(
            classification_path,
            epochs=100,
            unfreeze_schedule=custom_schedule,
            auto_optimize_confidence=True,  # Automatically optimize confidence after training
            overlap_handling_enabled=True,
        )

        print("\nStep 5: Saving enhanced model information...")
        classifier.save_enhanced_model_info()

        # Cleanup
        if os.path.exists("dataset/"):
            shutil.rmtree("dataset/")

        print("\n" + "=" * 60)
        print("ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return classifier

    except Exception as e:
        print(f"\nERROR: Enhanced training pipeline failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def example_enhanced_prediction(classifier, test_image_path):
    """
    Example of using the enhanced prediction capabilities.

    Args:
        classifier: Trained enhanced classifier
        test_image_path (str): Path to test image
    """
    print(f"\nEnhanced Prediction Example on: {test_image_path}")

    # Test different confidence types
    confidence_types = ["training", "validation", "inference", "strict"]

    for conf_type in confidence_types:
        print(f"\n--- Using {conf_type} confidence ---")
        result = classifier.predict_with_confidence_management(
            test_image_path, confidence_type=conf_type
        )

        if result and result["predictions"]:
            print(f"Confidence threshold used: {result['confidence_threshold_used']}")
            print(
                f"Top prediction: {result['top_prediction']['class_name']} ({result['top_prediction']['confidence']:.3f})"
            )
            print(f"All predictions: {len(result['predictions'])}")
        else:
            print("No predictions above threshold")

    # Test custom confidence
    print(f"\n--- Using custom confidence (0.3) ---")
    result = classifier.predict_with_confidence_management(test_image_path, custom_conf=0.3)

    if result and result["predictions"]:
        print(f"Predictions with custom threshold: {len(result['predictions'])}")
        for pred in result["predictions"]:
            print(f"  {pred['class_name']}: {pred['confidence']:.3f}")


if __name__ == "__main__":
    # Run the enhanced training pipeline
    trained_classifier = main_enhanced()
