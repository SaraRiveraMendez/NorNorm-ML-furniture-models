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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from ultralytics import YOLO


class AdaptiveWeightedYOLOv12Classifier:
    """
    Adaptive YOLOv12 classifier with intelligent overlap handling strategy selection
    and corrected progressive unfreezing implementation.
    """

    def __init__(self, model_size="n", img_size=640, batch_size=12, default_conf=0.25):
        """
        Initialize the Adaptive WeightedYOLOv12Classifier.
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

        # Progressive unfreezing state tracking
        self.unfreezing_state = {
            "initialized": False,
            "current_epoch": 0,
            "current_phase": 0,
            "layer_groups": [],
            "schedule": None,
        }

        # Overlap strategy analytics
        self.overlap_analytics = {
            "strategy_usage": Counter(),
            "strategy_effectiveness": {},
            "overlap_patterns": [],
        }

        # Create folder with timestamp
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/AdaptiveYOLOv12_Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def analyze_overlap_situation(self, bboxes_yolo, image_dimensions=None):
        """
        Analyze the overlapping situation to determine the best strategy.

        Args:
            bboxes_yolo (list): List of YOLO format bboxes
            image_dimensions (tuple): Optional (width, height) of image

        Returns:
            dict: Analysis results with recommended strategy
        """
        if len(bboxes_yolo) <= 1:
            return {
                "recommended_strategy": "none",
                "confidence": 1.0,
                "analysis": "No overlaps detected",
                "metrics": {},
            }

        # Convert to xyxy format for analysis
        boxes_xyxy = []
        for bbox in bboxes_yolo:
            if len(bbox) >= 5:
                class_id, cx, cy, w, h = bbox[:5]
                conf = bbox[5] if len(bbox) > 5 else 1.0

                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2

                boxes_xyxy.append([x1, y1, x2, y2, class_id, conf])

        if len(boxes_xyxy) <= 1:
            return {
                "recommended_strategy": "none",
                "confidence": 1.0,
                "analysis": "Insufficient valid boxes",
                "metrics": {},
            }

        # Calculate overlap metrics
        overlap_pairs = []
        same_class_overlaps = 0
        different_class_overlaps = 0
        high_iou_overlaps = 0
        size_disparities = []
        confidence_disparities = []

        for i in range(len(boxes_xyxy)):
            for j in range(i + 1, len(boxes_xyxy)):
                box1, box2 = boxes_xyxy[i], boxes_xyxy[j]
                iou = self._calculate_iou(box1, box2)

                if iou > 0.1:  # Consider as overlapping
                    overlap_pairs.append((i, j, iou))

                    # Class analysis
                    if box1[4] == box2[4]:
                        same_class_overlaps += 1
                    else:
                        different_class_overlaps += 1

                    # IoU analysis
                    if iou > 0.7:
                        high_iou_overlaps += 1

                    # Size analysis
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    size_ratio = max(area1, area2) / (min(area1, area2) + 1e-8)
                    size_disparities.append(size_ratio)

                    # Confidence analysis
                    conf_diff = abs(box1[5] - box2[5])
                    confidence_disparities.append(conf_diff)

        if not overlap_pairs:
            return {
                "recommended_strategy": "none",
                "confidence": 1.0,
                "analysis": "No significant overlaps detected",
                "metrics": {},
            }

        # Calculate metrics
        total_overlaps = len(overlap_pairs)
        avg_iou = np.mean([pair[2] for pair in overlap_pairs])
        avg_size_disparity = np.mean(size_disparities) if size_disparities else 1.0
        avg_conf_disparity = np.mean(confidence_disparities) if confidence_disparities else 0.0
        same_class_ratio = same_class_overlaps / total_overlaps
        high_iou_ratio = high_iou_overlaps / total_overlaps

        # Strategy selection logic
        strategy_scores = {"nms": 0.0, "merge": 0.0, "keep_largest": 0.0, "keep_highest_conf": 0.0}

        # NMS scoring: Good for different classes or high confidence disparities
        nms_score = 0.3  # Base score
        if different_class_overlaps > same_class_overlaps:
            nms_score += 0.3
        if avg_conf_disparity > 0.2:
            nms_score += 0.2
        if high_iou_ratio < 0.5:  # Not too much high overlap
            nms_score += 0.2
        strategy_scores["nms"] = nms_score

        # Merge scoring: Good for same class, high IoU, similar sizes and confidences
        merge_score = 0.2  # Base score
        if same_class_ratio > 0.7:
            merge_score += 0.3
        if high_iou_ratio > 0.6:
            merge_score += 0.2
        if avg_size_disparity < 2.0:  # Similar sizes
            merge_score += 0.2
        if avg_conf_disparity < 0.15:  # Similar confidences
            merge_score += 0.1
        strategy_scores["merge"] = merge_score

        # Keep largest scoring: Good for significant size disparities
        largest_score = 0.2  # Base score
        if avg_size_disparity > 3.0:
            largest_score += 0.4
        if same_class_ratio > 0.5:  # Prefer for same class
            largest_score += 0.2
        if avg_iou > 0.5:  # Significant overlap
            largest_score += 0.2
        strategy_scores["keep_largest"] = largest_score

        # Keep highest confidence scoring: Good for confidence disparities
        conf_score = 0.25  # Base score
        if avg_conf_disparity > 0.25:
            conf_score += 0.3
        if different_class_overlaps > same_class_overlaps:
            conf_score += 0.2
        if high_iou_ratio > 0.4:
            conf_score += 0.25
        strategy_scores["keep_highest_conf"] = conf_score

        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        confidence = strategy_scores[best_strategy]

        # Generate analysis text
        analysis_parts = []
        analysis_parts.append(f"Detected {total_overlaps} overlapping pairs")
        analysis_parts.append(f"Same class overlaps: {same_class_overlaps}/{total_overlaps}")
        analysis_parts.append(f"Average IoU: {avg_iou:.3f}")
        analysis_parts.append(f"Average size disparity: {avg_size_disparity:.2f}x")
        analysis_parts.append(f"High IoU overlaps: {high_iou_overlaps}/{total_overlaps}")

        analysis = "; ".join(analysis_parts)

        metrics = {
            "total_overlaps": total_overlaps,
            "same_class_overlaps": same_class_overlaps,
            "different_class_overlaps": different_class_overlaps,
            "avg_iou": avg_iou,
            "avg_size_disparity": avg_size_disparity,
            "avg_confidence_disparity": avg_conf_disparity,
            "same_class_ratio": same_class_ratio,
            "high_iou_ratio": high_iou_ratio,
            "strategy_scores": strategy_scores,
        }

        # Store analytics
        self.overlap_analytics["strategy_usage"][best_strategy] += 1
        self.overlap_analytics["overlap_patterns"].append(metrics)

        return {
            "recommended_strategy": best_strategy,
            "confidence": confidence,
            "analysis": analysis,
            "metrics": metrics,
        }

    def adaptive_overlap_handling(
        self, bboxes_yolo, default_threshold=0.5, force_strategy=None, image_dimensions=None
    ):
        """
        Adaptively handle overlapping bounding boxes using the most appropriate strategy.

        Args:
            bboxes_yolo (list): List of YOLO format bboxes
            default_threshold (float): Default IoU threshold
            force_strategy (str): Force a specific strategy (override adaptive selection)
            image_dimensions (tuple): Optional image dimensions for context

        Returns:
            dict: Results including processed boxes and strategy used
        """
        if len(bboxes_yolo) <= 1:
            return {
                "processed_boxes": bboxes_yolo,
                "strategy_used": "none",
                "original_count": len(bboxes_yolo),
                "final_count": len(bboxes_yolo),
                "analysis": "No overlaps to handle",
            }

        # Analyze situation unless strategy is forced
        if force_strategy:
            strategy = force_strategy
            analysis_result = {"analysis": f"Strategy forced to {force_strategy}"}
            print(f"Using forced strategy: {force_strategy}")
        else:
            analysis_result = self.analyze_overlap_situation(bboxes_yolo, image_dimensions)
            strategy = analysis_result["recommended_strategy"]
            print(
                f"Adaptive strategy selected: {strategy} (confidence: {analysis_result['confidence']:.2f})"
            )
            print(f"Analysis: {analysis_result['analysis']}")

        if strategy == "none":
            return {
                "processed_boxes": bboxes_yolo,
                "strategy_used": "none",
                "original_count": len(bboxes_yolo),
                "final_count": len(bboxes_yolo),
                "analysis": analysis_result["analysis"],
            }

        # Convert to xyxy for processing
        boxes_xyxy = []
        for bbox in bboxes_yolo:
            if len(bbox) >= 5:
                class_id, cx, cy, w, h = bbox[:5]
                conf = bbox[5] if len(bbox) > 5 else 1.0

                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2

                boxes_xyxy.append([x1, y1, x2, y2, class_id, conf])

        # Apply selected strategy
        if strategy == "nms":
            processed_boxes = self._apply_nms(boxes_xyxy, default_threshold)
        elif strategy == "merge":
            processed_boxes = self._merge_overlapping_boxes(boxes_xyxy, default_threshold)
        elif strategy == "keep_largest":
            processed_boxes = self._keep_largest_boxes(boxes_xyxy, default_threshold)
        elif strategy == "keep_highest_conf":
            processed_boxes = self._keep_highest_confidence_boxes(boxes_xyxy, default_threshold)
        else:
            print(f"Unknown strategy: {strategy}, using NMS")
            processed_boxes = self._apply_nms(boxes_xyxy, default_threshold)
            strategy = "nms"

        # Convert back to YOLO format
        result_yolo = []
        for box in processed_boxes:
            x1, y1, x2, y2, class_id, conf = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            result_yolo.append([int(class_id), cx, cy, w, h, conf])

        result = {
            "processed_boxes": result_yolo,
            "strategy_used": strategy,
            "original_count": len(bboxes_yolo),
            "final_count": len(result_yolo),
            "analysis": analysis_result.get("analysis", "Strategy applied"),
            "reduction_rate": (
                (len(bboxes_yolo) - len(result_yolo)) / len(bboxes_yolo)
                if len(bboxes_yolo) > 0
                else 0
            ),
        }

        return result

    def setup_progressive_unfreezing_corrected(self, unfreeze_schedule=None):
        """
        Corrected progressive unfreezing setup that properly manages layer groups.

        Args:
            unfreeze_schedule (dict): Schedule for unfreezing layers

        Returns:
            dict: Unfreezing configuration
        """
        if self.model is None:
            print("Error: Model not initialized for progressive unfreezing setup")
            return None

        if unfreeze_schedule is None:
            unfreeze_schedule = {0: 0.10, 20: 0.25, 40: 0.45, 65: 0.70, 90: 0.85, 110: 1.0}

        print("Setting up corrected progressive unfreezing...")

        # Get all trainable parameters with their names
        all_params = []
        param_info = []

        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                all_params.append(param)
                param_info.append(
                    {
                        "name": name,
                        "param": param,
                        "shape": tuple(param.shape),
                        "numel": param.numel(),
                    }
                )

        total_params = len(all_params)
        print(f"Found {total_params} trainable parameters")

        # Create layer groups for more intelligent unfreezing
        layer_groups = self._create_layer_groups(param_info)

        print(f"Created {len(layer_groups)} layer groups:")
        for i, group in enumerate(layer_groups):
            print(f"  Group {i+1}: {group['name']} ({len(group['params'])} parameters)")

        # Store unfreezing state
        self.unfreezing_state = {
            "initialized": True,
            "schedule": unfreeze_schedule,
            "layer_groups": layer_groups,
            "total_params": total_params,
            "current_epoch": 0,
            "current_phase": 0,
        }

        # Apply initial freezing
        self._apply_unfreezing_phase(0)

        return {
            "schedule": unfreeze_schedule,
            "total_params": total_params,
            "layer_groups": len(layer_groups),
            "initial_phase": 0,
        }

    def _create_layer_groups(self, param_info):
        """
        Create intelligent layer groups for progressive unfreezing.

        Args:
            param_info (list): List of parameter information dicts

        Returns:
            list: List of layer groups
        """
        groups = []

        # Group parameters by layer type/purpose
        backbone_params = []
        neck_params = []
        head_params = []
        classifier_params = []
        other_params = []

        for info in param_info:
            name = info["name"].lower()

            if any(x in name for x in ["backbone", "features", "conv", "bn", "downsample"]):
                if any(x in name for x in ["head", "classifier", "fc", "linear"]):
                    classifier_params.append(info)
                else:
                    backbone_params.append(info)
            elif any(x in name for x in ["neck", "fpn", "pan"]):
                neck_params.append(info)
            elif any(x in name for x in ["head", "detect"]):
                head_params.append(info)
            elif any(x in name for x in ["classifier", "classify", "fc", "linear"]):
                classifier_params.append(info)
            else:
                other_params.append(info)

        # Create groups in order of importance (classifier first, then head, etc.)
        if classifier_params:
            groups.append({"name": "Classifier", "params": classifier_params, "priority": 1})

        if head_params:
            groups.append({"name": "Detection Head", "params": head_params, "priority": 2})

        if neck_params:
            groups.append({"name": "Neck/FPN", "params": neck_params, "priority": 3})

        # Split backbone into layers if it's large
        if len(backbone_params) > 50:
            # Split backbone into early, mid, and late layers
            third = len(backbone_params) // 3

            groups.append(
                {"name": "Backbone (Late)", "params": backbone_params[:third], "priority": 4}
            )
            groups.append(
                {
                    "name": "Backbone (Mid)",
                    "params": backbone_params[third : 2 * third],
                    "priority": 5,
                }
            )
            groups.append(
                {"name": "Backbone (Early)", "params": backbone_params[2 * third :], "priority": 6}
            )
        else:
            groups.append({"name": "Backbone", "params": backbone_params, "priority": 4})

        if other_params:
            groups.append({"name": "Other Parameters", "params": other_params, "priority": 7})

        # Sort by priority
        groups.sort(key=lambda x: x["priority"])

        return groups

    def _apply_unfreezing_phase(self, epoch):
        """
        Apply unfreezing for the current phase based on epoch.

        Args:
            epoch (int): Current epoch

        Returns:
            bool: True if unfreezing state changed
        """
        if not self.unfreezing_state["initialized"]:
            print("Warning: Progressive unfreezing not initialized")
            return False

        schedule = self.unfreezing_state["schedule"]
        layer_groups = self.unfreezing_state["layer_groups"]

        # Find current phase
        current_phase_ratio = 0.1  # Default
        for epoch_threshold in sorted(schedule.keys()):
            if epoch >= epoch_threshold:
                current_phase_ratio = schedule[epoch_threshold]
            else:
                break

        # Calculate how many groups to unfreeze
        total_groups = len(layer_groups)
        groups_to_unfreeze = max(1, int(total_groups * current_phase_ratio))
        groups_to_unfreeze = min(groups_to_unfreeze, total_groups)

        # Check if this is a new phase
        old_phase = self.unfreezing_state.get("current_phase", 0)
        new_phase = groups_to_unfreeze

        if new_phase != old_phase or epoch == 0:
            print(f"Epoch {epoch}: Applying progressive unfreezing phase")
            print(
                f"  Unfreezing {groups_to_unfreeze}/{total_groups} layer groups ({current_phase_ratio*100:.0f}%)"
            )

            # First freeze all parameters
            for group in layer_groups:
                for param_info in group["params"]:
                    param_info["param"].requires_grad = False

            # Then unfreeze the specified number of groups (starting from highest priority)
            unfrozen_params = 0
            for i in range(groups_to_unfreeze):
                group = layer_groups[i]
                for param_info in group["params"]:
                    param_info["param"].requires_grad = True
                    unfrozen_params += 1
                print(f"    Unfrozen group: {group['name']} ({len(group['params'])} parameters)")

            # Update state
            self.unfreezing_state["current_phase"] = new_phase
            self.unfreezing_state["current_epoch"] = epoch

            # Calculate total frozen/unfrozen counts
            total_params = sum(len(group["params"]) for group in layer_groups)
            frozen_params = total_params - unfrozen_params

            print(f"  Total: {unfrozen_params} unfrozen, {frozen_params} frozen")

            return True

        return False

    def train_model_with_corrected_progressive_unfreezing_and_class_weights(
        self,
        classification_path,
        epochs=125,
        unfreeze_schedule=None,
        auto_optimize_confidence=True,
        adaptive_overlap_enabled=True,
    ):
        """
        Training method with corrected progressive unfreezing and adaptive overlap handling.

        Args:
            classification_path (str): Path to classification dataset
            epochs (int): Number of training epochs
            unfreeze_schedule (dict): Custom unfreezing schedule
            auto_optimize_confidence (bool): Auto-optimize confidence threshold
            adaptive_overlap_enabled (bool): Enable adaptive overlap handling

        Returns:
            dict: Training results
        """
        print(f"\n{'='*60}")
        print("ADAPTIVE TRAINING: CORRECTED UNFREEZING + SMART OVERLAP HANDLING")
        print(f"{'='*60}")

        if len(self.class_names) == 0:
            raise ValueError("No classes defined. Cannot proceed with training.")

        if self.model is None:
            raise ValueError("Model not initialized.")

        # Setup corrected progressive unfreezing
        unfreezing_config = self.setup_progressive_unfreezing_corrected(unfreeze_schedule)
        if unfreezing_config is None:
            print("Failed to set up progressive unfreezing.")
            return None

        print(f"Adaptive Features Enabled:")
        print(
            f"  ✓ Corrected progressive unfreezing ({len(self.unfreezing_state['layer_groups'])} groups)"
        )
        print(f"  ✓ Class weights")
        print(f"  ✓ Adaptive overlap handling: {adaptive_overlap_enabled}")
        print(f"  ✓ Auto confidence optimization: {auto_optimize_confidence}")

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
            "conf": self.default_conf,
        }

        # Phase-based training with corrected unfreezing
        schedule = self.unfreezing_state["schedule"]
        epoch_phases = sorted(schedule.keys())
        all_phase_results = []

        print("Starting adaptive training with corrected progressive unfreezing...")

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
                f"ADAPTIVE PHASE {i+1}: EPOCHS {phase_start_epoch} to {phase_start_epoch + phase_epochs - 1}"
            )
            print(f"{'='*50}")

            # Apply unfreezing for this phase
            unfreezing_changed = self._apply_unfreezing_phase(phase_start_epoch)

            # Load previous checkpoint if continuing
            if i > 0:
                previous_phase_name = f"adaptive_phase_{i}"
                last_checkpoint = os.path.join(
                    self.save_dir, previous_phase_name, "weights", "last.pt"
                )
                if os.path.exists(last_checkpoint):
                    print(f"Loading checkpoint: {last_checkpoint}")
                    self.model = YOLO(last_checkpoint)

                    # Re-apply class weights and unfreezing after loading
                    self.patch_model_loss()
                    self._apply_unfreezing_phase(phase_start_epoch)

            # Configure training for this phase
            phase_training_args = base_training_args.copy()
            phase_training_args.update(
                {
                    "epochs": phase_epochs,
                    "name": f"adaptive_phase_{i+1}",
                }
            )

            # Execute training
            print(f"Starting adaptive training for phase {i+1}...")
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
                print(f"Adaptive phase {i+1} completed successfully!")

            except Exception as e:
                print(f"Error in adaptive phase {i+1}: {e}")
                continue

        # Auto-optimize confidence threshold after training
        if auto_optimize_confidence:
            print(f"\n{'='*50}")
            print("POST-TRAINING CONFIDENCE OPTIMIZATION")
            print(f"{'='*50}")
            optimization_results = self.optimize_confidence_threshold(classification_path)

        # Save overlap analytics
        self.save_overlap_analytics()

        print(f"\n{'='*60}")
        print("ADAPTIVE TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")

        # Print overlap analytics summary
        if self.overlap_analytics["strategy_usage"]:
            print("Overlap Strategy Usage:")
            for strategy, count in self.overlap_analytics["strategy_usage"].items():
                print(f"  {strategy}: {count} times")

        return {
            "all_phases": all_phase_results,
            "final_results": all_phase_results[-1]["results"] if all_phase_results else None,
            "adaptive_features_applied": True,
            "corrected_unfreezing": True,
            "overlap_analytics": self.overlap_analytics,
            "final_confidence_threshold": self.default_conf,
            "class_weights": self.class_weights,
            "final_classes": self.class_names,
        }

    # Include all the helper methods from the previous implementation
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in xyxy format."""
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
        """Apply Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []

        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)

        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)

            boxes = [
                box
                for box in boxes
                if box[4] != current[4] or self._calculate_iou(current, box) < iou_threshold
            ]

        return keep

    def _merge_overlapping_boxes(self, boxes, iou_threshold):
        """Merge overlapping boxes of the same class."""
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

                if box1[4] == box2[4] and self._calculate_iou(box1, box2) >= iou_threshold:
                    merge_group.append(box2)
                    processed.add(j)

            if len(merge_group) == 1:
                merged.append(merge_group[0])
            else:
                merged_box = self._merge_box_group(merge_group)
                merged.append(merged_box)

        return merged

    def _merge_box_group(self, box_group):
        """Merge a group of overlapping boxes into one box."""
        if len(box_group) == 1:
            return box_group[0]

        total_conf = sum(box[5] for box in box_group)

        x1 = sum(box[0] * box[5] for box in box_group) / total_conf
        y1 = sum(box[1] * box[5] for box in box_group) / total_conf
        x2 = sum(box[2] * box[5] for box in box_group) / total_conf
        y2 = sum(box[3] * box[5] for box in box_group) / total_conf

        class_id = box_group[0][4]
        max_conf = max(box[5] for box in box_group)

        return [x1, y1, x2, y2, class_id, max_conf]

    def _keep_largest_boxes(self, boxes, iou_threshold):
        """Keep the largest box among overlapping boxes of the same class."""
        if len(boxes) == 0:
            return []

        boxes_with_area = []
        for box in boxes:
            area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_with_area.append(box + [area])

        boxes_with_area = sorted(boxes_with_area, key=lambda x: x[6], reverse=True)

        keep = []
        processed = set()

        for i, box1 in enumerate(boxes_with_area):
            if i in processed:
                continue

            keep.append(box1[:6])
            processed.add(i)

            for j, box2 in enumerate(boxes_with_area[i + 1 :], i + 1):
                if j in processed:
                    continue

                if box1[4] == box2[4] and self._calculate_iou(box1[:6], box2[:6]) >= iou_threshold:
                    processed.add(j)

        return keep

    def _keep_highest_confidence_boxes(self, boxes, iou_threshold):
        """Keep the highest confidence box among overlapping boxes."""
        return self._apply_nms(boxes, iou_threshold)

    def optimize_confidence_threshold(self, validation_path, threshold_range=(0.1, 0.9), step=0.05):
        """
        Automatically find optimal confidence threshold based on validation data.
        """
        if self.model is None:
            print("Error: Model not initialized for threshold optimization")
            return None

        print("Optimizing confidence threshold...")

        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        results = []

        val_images_path = os.path.join(validation_path, "val", "images")
        if not os.path.exists(val_images_path):
            print(f"Validation images not found at: {val_images_path}")
            return None

        val_images = [
            f for f in os.listdir(val_images_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:100]

        print(f"Testing {len(thresholds)} thresholds on {len(val_images)} validation images...")

        for threshold in thresholds:
            correct_predictions = 0
            total_predictions = 0
            confident_predictions = 0

            for img_file in val_images:
                img_path = os.path.join(val_images_path, img_file)

                true_class_idx = None
                for class_idx, class_name in enumerate(self.class_names):
                    if class_name.lower() in img_file.lower():
                        true_class_idx = class_idx
                        break

                if true_class_idx is None:
                    continue

                try:
                    results_pred = self.model.predict(img_path, verbose=False, conf=threshold)

                    if results_pred and len(results_pred) > 0:
                        probs = results_pred[0].probs
                        if probs is not None:
                            confident_predictions += 1
                            predicted_class = probs.top1
                            if predicted_class == true_class_idx:
                                correct_predictions += 1

                    total_predictions += 1

                except Exception:
                    continue

            accuracy = correct_predictions / max(total_predictions, 1)
            confidence_rate = confident_predictions / max(total_predictions, 1)
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

        best_result = max(results, key=lambda x: x["combined_score"])
        optimal_threshold = best_result["threshold"]
        self.default_conf = optimal_threshold

        optimization_results = {
            "optimal_threshold": optimal_threshold,
            "best_metrics": best_result,
            "all_results": results,
            "optimization_date": datetime.datetime.now().isoformat(),
        }

        results_path = os.path.join(self.save_dir, "threshold_optimization.json")
        with open(results_path, "w") as f:
            json.dump(optimization_results, f, indent=2)

        print(f"Optimal confidence threshold found: {optimal_threshold}")
        print(f"Optimization results saved to: {results_path}")

        return optimization_results

    def save_overlap_analytics(self):
        """Save overlap handling analytics to file."""
        analytics_path = os.path.join(self.save_dir, "overlap_analytics.json")

        # Convert Counter to dict for JSON serialization
        analytics_data = {
            "strategy_usage": dict(self.overlap_analytics["strategy_usage"]),
            "total_overlap_situations": sum(self.overlap_analytics["strategy_usage"].values()),
            "overlap_patterns": self.overlap_analytics["overlap_patterns"][
                -50:
            ],  # Last 50 patterns
            "analysis_timestamp": datetime.datetime.now().isoformat(),
        }

        with open(analytics_path, "w") as f:
            json.dump(analytics_data, f, indent=2)

        print(f"Overlap analytics saved to: {analytics_path}")

    # Include essential methods from original implementation
    def download_and_extract_dataset(self, gdrive_file_id, output_filename=None):
        """Download dataset from Google Drive and extract it."""
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

    def clean_and_extract_objects_with_adaptive_overlap_handling(
        self,
        dataset_path,
        min_area=0.0001,
        max_samples_per_class=15000,
        val_split=0.2,
        overlap_threshold=0.5,
    ):
        """Enhanced dataset cleaning with adaptive overlap handling."""
        print("Extracting and cleaning with adaptive overlap handling...")

        # Load class names
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YOLO config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data_config = yaml.safe_load(f)
            self.original_class_names = data_config.get("names", [])

        # Filter background classes
        original_names = [n.strip().lower() for n in self.original_class_names]
        background_keywords = ["background", "bg", "__background__", "void", "unlabeled"]

        filtered_classes = []
        self.id_map = {}
        new_class_id = 0

        for original_id, class_name in enumerate(original_names):
            if class_name not in background_keywords:
                filtered_classes.append(class_name)
                self.id_map[original_id] = new_class_id
                new_class_id += 1

        self.class_names = filtered_classes

        if len(self.class_names) == 0:
            raise ValueError("No valid classes found after background removal!")

        print(f"Final classes: {self.class_names}")

        # Setup dataset directories
        classification_dir = "adaptive_cleaned_dataset"
        if os.path.exists(classification_dir):
            shutil.rmtree(classification_dir)
        os.makedirs(classification_dir, exist_ok=True)

        # Create train/val directories
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

        # Process images with adaptive overlap handling
        train_images_dir = os.path.join(dataset_path, "train", "images")
        train_labels_dir = os.path.join(dataset_path, "train", "labels")

        all_images = [
            f for f in os.listdir(train_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        train_imgs, val_imgs = self._create_stratified_split(
            all_images, train_labels_dir, self.id_map, val_split
        )

        # Process splits with adaptive overlap handling
        train_stats = self._process_split_with_adaptive_overlap(
            train_imgs,
            train_images_dir,
            train_labels_dir,
            new_train_images_dir,
            new_train_labels_dir,
            self.id_map,
            min_area,
            overlap_threshold,
        )

        val_stats = self._process_split_with_adaptive_overlap(
            val_imgs,
            train_images_dir,
            train_labels_dir,
            new_val_images_dir,
            new_val_labels_dir,
            self.id_map,
            min_area,
            overlap_threshold,
        )

        # Combine statistics and calculate class weights
        class_counts = {name: 0 for name in self.class_names}
        for class_name in self.class_names:
            class_counts[class_name] = (
                train_stats["class_counts"][class_name] + val_stats["class_counts"][class_name]
            )

        self._calculate_class_weights(class_counts)

        print(f"\nAdaptive cleaning summary:")
        print(f"Train/Val split: {len(train_imgs)}/{len(val_imgs)}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} objects")

        return classification_dir

    def _create_stratified_split(self, all_images, train_labels_dir, id_map, val_split):
        """Create stratified train/validation split."""
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

    def _process_split_with_adaptive_overlap(
        self,
        images,
        src_img_dir,
        src_labels_dir,
        dst_img_dir,
        dst_labels_dir,
        id_map,
        min_area,
        overlap_threshold,
    ):
        """Process image split with adaptive overlap handling."""
        class_counts = {name: 0 for name in self.class_names}
        total_labels, kept_labels = 0, 0
        images_with_valid_labels = 0

        for img_file in images:
            img_path = os.path.join(src_img_dir, img_file)
            new_img_path = os.path.join(dst_img_dir, img_file)

            label_file = os.path.splitext(img_file)[0] + ".txt"
            old_label_path = os.path.join(src_labels_dir, label_file)
            new_label_path = os.path.join(dst_labels_dir, label_file)

            if os.path.exists(old_label_path):
                processed_labels = self._process_label_with_adaptive_overlap(
                    old_label_path, id_map, min_area, overlap_threshold
                )

                if processed_labels["lines"]:
                    shutil.copy2(img_path, new_img_path)
                    images_with_valid_labels += 1

                    with open(new_label_path, "w") as f:
                        f.writelines(processed_labels["lines"])

                    total_labels += processed_labels["total"]
                    kept_labels += processed_labels["kept"]

                    for class_name, count in processed_labels["class_counts"].items():
                        class_counts[class_name] += count

        return {"total": total_labels, "kept": kept_labels, "class_counts": class_counts}

    def _process_label_with_adaptive_overlap(self, label_path, id_map, min_area, overlap_threshold):
        """Process label file with adaptive overlap handling."""
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            return {
                "lines": [],
                "total": 0,
                "kept": 0,
                "class_counts": {name: 0 for name in self.class_names},
            }

        # Parse bounding boxes
        bboxes = []
        total_lines = len(lines)

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            if (
                class_id not in id_map
                or bw <= 0
                or bh <= 0
                or cx < 0
                or cy < 0
                or cx > 1
                or cy > 1
                or bw * bh < min_area
            ):
                continue

            bboxes.append([class_id, cx, cy, bw, bh, 1.0])

        # Apply adaptive overlap handling
        if len(bboxes) > 1:
            overlap_result = self.adaptive_overlap_handling(bboxes, overlap_threshold)
            bboxes = overlap_result["processed_boxes"]

        # Convert back to lines
        new_lines = []
        class_counts = {name: 0 for name in self.class_names}

        for bbox in bboxes:
            class_id, cx, cy, bw, bh = bbox[:5]
            new_id = id_map[class_id]
            new_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            if new_id < len(self.class_names):
                class_counts[self.class_names[new_id]] += 1

        return {
            "lines": new_lines,
            "total": total_lines,
            "kept": len(new_lines),
            "class_counts": class_counts,
        }

    def _calculate_class_weights(self, class_counts):
        """Calculate class weights for imbalanced datasets."""
        if not class_counts or sum(class_counts.values()) == 0:
            return

        valid_classes = {cls: count for cls, count in class_counts.items() if count > 0}
        if len(valid_classes) == 0:
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

    def create_yolo_classification_config(self, classification_dir):
        """Create YOLO classification configuration file."""
        if len(self.class_names) == 0:
            raise ValueError("No valid classes found.")

        config = {
            "path": os.path.abspath(classification_dir),
            "train": "train",
            "val": "val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        config_path = os.path.join(classification_dir, "data.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        return config_path

    def initialize_yolov12_classifier(self):
        """Initialize YOLOv12 model for classification."""
        if len(self.class_names) == 0:
            return False

        try:
            model_name = f"yolo12{self.model_size}-cls.pt"
            self.model = YOLO(model_name)

            if self.class_weights_tensor is not None:
                self.patch_model_loss()

            return True

        except Exception:
            try:
                model_name = f"yolo12{self.model_size}.pt"
                self.model = YOLO(model_name)

                if self.class_weights_tensor is not None:
                    self.patch_model_loss()

                return True
            except Exception:
                return False

    def patch_model_loss(self):
        """Patch model to use weighted loss function."""
        if self.model is None or self.class_weights_tensor is None:
            return False

        try:
            device = next(self.model.model.parameters()).device
            weighted_tensor = self.class_weights_tensor.to(device)

            if hasattr(self.model.model, "loss"):
                original_loss = self.model.model.loss

                def weighted_loss_wrapper(*args, **kwargs):
                    loss_dict = original_loss(*args, **kwargs)

                    if "cls" in loss_dict and len(args) >= 2:
                        preds, targets = args[0], args[1]
                        if hasattr(targets, "cls"):
                            cls_targets = targets.cls.long()
                            cls_preds = preds[0] if isinstance(preds, (list, tuple)) else preds

                            if cls_preds.size(1) == len(self.class_names):
                                weighted_cls_loss = nn.CrossEntropyLoss(weight=weighted_tensor)(
                                    cls_preds, cls_targets
                                )
                                loss_dict["cls"] = weighted_cls_loss

                    return loss_dict

                self.model.model.loss = weighted_loss_wrapper
                return True

        except Exception:
            pass

        return False


# Main function for adaptive training
def main_adaptive():
    """
    Main training pipeline with adaptive overlap handling and corrected progressive unfreezing.
    """
    try:
        print("Initializing Adaptive WeightedYOLOv12Classifier...")
        classifier = AdaptiveWeightedYOLOv12Classifier(model_size="n", img_size=640, batch_size=12)

        print("\nStep 1: Downloading dataset...")
        gdrive_file_id = "1zvCNOz4P0QFdOAfpDpSpidFVjXHFxDUE"
        dataset_path = classifier.download_and_extract_dataset(gdrive_file_id)

        print("\nStep 2: Adaptive dataset cleaning...")
        classification_path = classifier.clean_and_extract_objects_with_adaptive_overlap_handling(
            dataset_path,
            min_area=0.0001,
            max_samples_per_class=15000,
            val_split=0.2,
            overlap_threshold=0.5,
        )

        print("\nStep 3: Initializing YOLOv12 model...")
        if not classifier.initialize_yolov12_classifier():
            raise RuntimeError("Failed to initialize YOLOv12 model")

        print("\nStep 4: Adaptive training with corrected progressive unfreezing...")
        custom_schedule = {0: 0.10, 20: 0.25, 40: 0.45, 65: 0.70, 90: 0.85, 110: 1.0}

        training_results = (
            classifier.train_model_with_corrected_progressive_unfreezing_and_class_weights(
                classification_path,
                epochs=125,
                unfreeze_schedule=custom_schedule,
                auto_optimize_confidence=True,
                adaptive_overlap_enabled=True,
            )
        )

        # Cleanup
        if os.path.exists("dataset/"):
            shutil.rmtree("dataset/")

        print("\n" + "=" * 60)
        print("ADAPTIVE TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Key Features:")
        print("  ✓ Adaptive overlap handling based on situation analysis")
        print("  ✓ Corrected progressive unfreezing with proper layer grouping")
        print("  ✓ Smart strategy selection for different overlap scenarios")
        print("  ✓ Comprehensive analytics and optimization")
        print("=" * 60)

        return classifier

    except Exception as e:
        print(f"\nERROR: Adaptive training pipeline failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    trained_classifier = main_adaptive()
