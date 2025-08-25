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
import tensorflow as tf
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from ultralytics import YOLO


class ImprovedFurnitureModelYOLOv12:
    def __init__(self, img_size=(640, 640), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_names = []
        self.yolo_model = None

        # Create folder with timestamp for this training session
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.save_dir = f"Models/Model({timestamp})"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def download_and_extract_remote_dataset(self, gdrive_file_id, output_filename=None):
        """Download dataset from Google Drive and extract without storing zip permanently"""
        import tempfile

        if output_filename is None:
            output_filename = f"dataset_{gdrive_file_id}.zip"

        url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        # Use temporary directory to avoid storage overhead
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, output_filename)

            print("Downloading dataset from Google Drive...")
            gdown.download(url, temp_zip_path, quiet=False)

            # Extract to permanent location
            extract_path = "dataset/"
            print("Extracting dataset...")
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

        return extract_path

    def parse_yolo_annotations_with_split(self, dataset_path, train_split=0.8):
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

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            image = image.astype("float32") / 255.0
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def create_data_generators(self, train_data, val_data):
        """Create data generators without augmentation since dataset already includes it"""
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Labels are already in categorical format from the dataset
        num_classes = len(self.class_names)
        y_train_cat = train_labels  # Already categorical
        y_val_cat = val_labels  # Already categorical

        class ImprovedSequence(tf.keras.utils.Sequence):
            def __init__(self, image_paths, labels, batch_size, img_size, shuffle=True):
                self.image_paths = image_paths
                self.labels = labels
                self.batch_size = batch_size
                self.img_size = img_size
                self.shuffle = shuffle
                self.indices = np.arange(len(self.image_paths))
                self.on_epoch_end()

            def __len__(self):
                return max(1, len(self.image_paths) // self.batch_size)

            def __getitem__(self, index):
                batch_indices = self.indices[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]

                batch_images = []
                batch_labels = []

                for idx in batch_indices:
                    if idx < len(self.image_paths):
                        img_path = self.image_paths[idx]
                        label = self.labels[idx]

                        image = self.load_and_preprocess_image_batch(img_path)
                        if image is not None:
                            batch_images.append(image)
                            batch_labels.append(label)

                # Convert to numpy arrays
                batch_images = np.array(batch_images)
                batch_labels = np.array(batch_labels)

                # Ensure labels are in the correct format (categorical)
                if len(batch_labels.shape) == 1:
                    batch_labels = tf.keras.utils.to_categorical(
                        batch_labels, num_classes=num_classes
                    )

                return batch_images, batch_labels

            def load_and_preprocess_image_batch(self, image_path):
                """Load and preprocess image for batch processing"""
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        return None
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.img_size)
                    image = image.astype("float32") / 255.0
                    return image
                except Exception:
                    return None

            def on_epoch_end(self):
                if self.shuffle:
                    np.random.shuffle(self.indices)

        train_generator = ImprovedSequence(
            train_images, y_train_cat, self.batch_size, self.img_size, shuffle=True
        )
        val_generator = ImprovedSequence(
            val_images, y_val_cat, self.batch_size, self.img_size, shuffle=False
        )

        # Store generators for confusion matrix calculation
        self.train_generator = train_generator
        self.val_generator = val_generator

        return train_generator, val_generator

    def initialize_yolov12(self, model_size="n"):
        """Initialize YOLOv12 model"""
        try:
            model_name = f"yolo12{model_size}.pt"
            self.yolo_model = YOLO(model_name)
            print(f"YOLOv12{model_size} initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing YOLOv12: {e}")
            print("Please ensure ultralytics is installed and YOLOv12 model is available")
            return False

    def extract_yolo_features(self, image_batch):
        """Extract features directly from YOLO backbone"""
        if self.yolo_model is None:
            return None

        features_list = []
        for img in image_batch:
            img_uint8 = (img * 255).astype(np.uint8)

            img_tensor = self.yolo_model.transforms(img_uint8)
            img_tensor = img_tensor.unsqueeze(0).to(self.yolo_model.device)

            with torch.no_grad():
                features = self.yolo_model.model.backbone(img_tensor)
                last_feature_map = features[-1]  # Último feature map

                # Global Average Pooling
                pooled = torch.mean(last_feature_map, dim=[2, 3]).squeeze().cpu().numpy()
                features_list.append(pooled)

        return np.array(features_list)

    def build_model(self, num_classes):
        """Build model with REAL YOLOv12 integration"""

        # INPUT LAYER
        input_layer = tf.keras.layers.Input(shape=(*self.img_size, 3))

        # YOLO BACKBONE INTEGRATION
        if self.yolo_model is not None:
            print("Building model with YOLOv12 backbone integration")

            # Custom layer for YOLO feature extraction
            class YOLOFeatureExtractor(tf.keras.layers.Layer):
                def __init__(self, yolo_model, feature_dim=2048, **kwargs):
                    super(YOLOFeatureExtractor, self).__init__(**kwargs)
                    self.yolo_model = yolo_model
                    self.feature_dim = feature_dim
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # Pre-warm the model to avoid initialization issues
                    self._warmup_model()

                def _warmup_model(self):
                    """Warm up the model to avoid initialization issues"""
                    try:
                        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                        with torch.no_grad():
                            # Try different ways to access the model
                            if (
                                hasattr(self.yolo_model, "model")
                                and self.yolo_model.model is not None
                            ):
                                _ = self.yolo_model.model(dummy_input)
                            elif (
                                hasattr(self.yolo_model, "predictor")
                                and self.yolo_model.predictor is not None
                            ):
                                if hasattr(self.yolo_model.predictor, "model"):
                                    _ = self.yolo_model.predictor.model(dummy_input)
                    except Exception as e:
                        print(f"Warning during model warmup: {e}")

                def build(self, input_shape):
                    super(YOLOFeatureExtractor, self).build(input_shape)

                def call(self, inputs):
                    def extract_features_batch(batch_images):
                        batch_features = []
                        batch_np = batch_images.numpy()

                        for i in range(batch_np.shape[0]):
                            img_np = batch_np[i]

                            try:
                                # Enhanced method with better YOLO model access
                                yolo_internal_model = None

                                # Try multiple ways to access the YOLO model
                                if (
                                    hasattr(self.yolo_model, "model")
                                    and self.yolo_model.model is not None
                                ):
                                    yolo_internal_model = self.yolo_model.model
                                elif (
                                    hasattr(self.yolo_model, "predictor")
                                    and self.yolo_model.predictor is not None
                                ):
                                    if hasattr(self.yolo_model.predictor, "model"):
                                        yolo_internal_model = self.yolo_model.predictor.model

                                if yolo_internal_model is not None and hasattr(
                                    yolo_internal_model, "backbone"
                                ):
                                    # Method 1: Direct feature extraction from backbone
                                    # Preprocess image for YOLO
                                    img_tensor = (
                                        torch.from_numpy(img_np)
                                        .permute(2, 0, 1)
                                        .unsqueeze(0)
                                        .float()
                                    )
                                    img_tensor = img_tensor.to(self.device)
                                    img_tensor = img_tensor / 255.0

                                    # Resize to expected input size
                                    if img_tensor.shape[2:] != (640, 640):
                                        img_tensor = torch.nn.functional.interpolate(
                                            img_tensor,
                                            size=(640, 640),
                                            mode="bilinear",
                                            align_corners=False,
                                        )

                                    # Extract features using backbone only
                                    with torch.no_grad():
                                        features = yolo_internal_model.backbone(img_tensor)

                                        # Handle different feature output formats
                                        if isinstance(features, (list, tuple)):
                                            # Use the last feature map
                                            last_feature = features[-1]
                                        elif isinstance(features, dict):
                                            # Handle dictionary output (common in some YOLO versions)
                                            last_feature = list(features.values())[-1]
                                        else:
                                            last_feature = features

                                        # Global average pooling
                                        pooled_features = torch.mean(
                                            last_feature, dim=[2, 3]
                                        ).squeeze()
                                        feature_vector = pooled_features.cpu().numpy()

                                else:
                                    # Method 2: Alternative approach using model's predict method
                                    # Use YOLO's built-in prediction but extract intermediate features
                                    results = self.yolo_model.predict(
                                        img_np, verbose=False, save=False, augment=False
                                    )

                                    # Create feature vector from detection results
                                    if results and len(results) > 0:
                                        result = results[0]
                                        boxes = result.boxes

                                        if boxes is not None and len(boxes) > 0:
                                            confidences = boxes.conf.cpu().numpy()
                                            num_detections = len(boxes)
                                            avg_confidence = (
                                                float(np.mean(confidences))
                                                if len(confidences) > 0
                                                else 0.0
                                            )
                                            max_confidence = (
                                                float(np.max(confidences))
                                                if len(confidences) > 0
                                                else 0.0
                                            )
                                            min_confidence = (
                                                float(np.min(confidences))
                                                if len(confidences) > 0
                                                else 0.0
                                            )

                                            # More comprehensive feature vector
                                            feature_vector = np.array(
                                                [
                                                    num_detections,
                                                    avg_confidence,
                                                    max_confidence,
                                                    min_confidence,
                                                    (
                                                        float(np.std(confidences))
                                                        if len(confidences) > 1
                                                        else 0.0
                                                    ),
                                                    *([0.0] * (self.feature_dim - 5)),
                                                ]
                                            ).astype(np.float32)
                                        else:
                                            feature_vector = np.zeros(
                                                self.feature_dim, dtype=np.float32
                                            )
                                    else:
                                        feature_vector = np.zeros(
                                            self.feature_dim, dtype=np.float32
                                        )

                                # Ensure consistent feature dimension
                                feature_vector = feature_vector.flatten()
                                if len(feature_vector) < self.feature_dim:
                                    feature_vector = np.pad(
                                        feature_vector,
                                        (0, self.feature_dim - len(feature_vector)),
                                        mode="constant",
                                    )
                                elif len(feature_vector) > self.feature_dim:
                                    feature_vector = feature_vector[: self.feature_dim]

                            except Exception as e:
                                print(f"Error in YOLO feature extraction for image {i}: {e}")
                                feature_vector = np.zeros(self.feature_dim, dtype=np.float32)

                            batch_features.append(feature_vector.astype(np.float32))

                        return np.array(batch_features, dtype=np.float32)

                    # Use tf.py_function to process the batch
                    features = tf.py_function(
                        func=extract_features_batch, inp=[inputs], Tout=tf.float32
                    )

                    # Set the shape explicitly
                    features.set_shape([None, self.feature_dim])

                    return features

                def get_config(self):
                    config = super(YOLOFeatureExtractor, self).get_config()
                    config.update({"feature_dim": self.feature_dim})
                    return config

            # Extract YOLO features with smaller feature dimension to avoid issues
            yolo_features = YOLOFeatureExtractor(self.yolo_model, feature_dim=1024)(input_layer)

            # Build classification head
            x = layers.Dense(512, activation="relu", kernel_regularizer=l2(0.001))(yolo_features)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)

            x = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

            x = layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

            # Output layer
            outputs = layers.Dense(num_classes, activation="softmax")(x)

            # Create model
            model = tf.keras.Model(inputs=input_layer, outputs=outputs)

        else:
            print("Building model with EfficientNetB0 backbone (YOLO not available)")

            # Fallback to EfficientNetB0 if YOLO is not available
            base_model = tf.keras.applications.EfficientNetB0(
                weights="imagenet", include_top=False, input_tensor=input_layer
            )

            base_model.trainable = True

            # Freeze most layers initially, keep only last 30 trainable
            fine_tune_at = len(base_model.layers) - 30
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

            # Complete architecture
            model = models.Sequential(
                [
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
                    layers.BatchNormalization(),
                    layers.Dropout(0.4),
                    layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        # Compilation (same as before)
        initial_learning_rate = 5e-5 if self.yolo_model is not None else 1e-4

        class Top1Accuracy(tf.keras.metrics.Metric):
            def __init__(self, name="top_1_accuracy", **kwargs):
                super(Top1Accuracy, self).__init__(name=name, **kwargs)
                self.total = self.add_weight(name="total", initializer="zeros")
                self.count = self.add_weight(name="count", initializer="zeros")

            def update_state(self, y_true, y_pred, sample_weight=None):
                predictions = tf.argmax(y_pred, axis=1)
                targets = tf.argmax(y_true, axis=1)
                correct = tf.cast(tf.equal(predictions, targets), tf.float32)
                self.total.assign_add(tf.reduce_sum(correct))
                self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

            def result(self):
                return self.total / self.count

            def reset_state(self):
                self.total.assign(0.0)
                self.count.assign(0.0)

        metrics = ["accuracy", Top1Accuracy()]

        try:
            top3_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
            metrics.append(top3_metric)
        except AttributeError:
            print("Top-3 accuracy not available in this TensorFlow version")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        self.model = model
        if self.yolo_model is None and "base_model" in locals():
            self.base_model = base_model

        return model

    def setup_progressive_unfreezing(self):
        """Setup progressive unfreezing for YOLO model"""
        if self.yolo_model is not None:
            # Enhanced YOLO model access
            yolo_internal_model = None

            # Try multiple ways to access the internal model
            if hasattr(self.yolo_model, "model") and self.yolo_model.model is not None:
                yolo_internal_model = self.yolo_model.model
            elif hasattr(self.yolo_model, "predictor") and self.yolo_model.predictor is not None:
                if hasattr(self.yolo_model.predictor, "model"):
                    yolo_internal_model = self.yolo_model.predictor.model

            if yolo_internal_model is not None:
                # Freeze all YOLO layers initially
                for param in yolo_internal_model.parameters():
                    param.requires_grad = False

                print(
                    f"YOLO model frozen initially. Total parameters: {sum(p.numel() for p in yolo_internal_model.parameters())}"
                )
                return yolo_internal_model
            else:
                print("Could not access YOLO internal model for progressive unfreezing")
                return None
        return None

    def unfreeze_yolo_layers(self, epoch, total_epochs):
        """Progressively unfreeze YOLO layers during training"""
        if self.yolo_model is None:
            return

        # Enhanced YOLO model access
        yolo_internal_model = None

        if hasattr(self.yolo_model, "model") and self.yolo_model.model is not None:
            yolo_internal_model = self.yolo_model.model
        elif hasattr(self.yolo_model, "predictor") and self.yolo_model.predictor is not None:
            if hasattr(self.yolo_model.predictor, "model"):
                yolo_internal_model = self.yolo_model.predictor.model

        if yolo_internal_model is None:
            return

        # Define unfreezing schedule
        unfreeze_schedule = {
            int(total_epochs * 0.3): "last_layers",  # 30% through: unfreeze last few layers
            int(total_epochs * 0.5): "middle_layers",  # 50% through: unfreeze middle layers
            int(total_epochs * 0.7): "early_layers",  # 70% through: unfreeze early layers
            int(total_epochs * 0.9): "all_layers",  # 90% through: unfreeze all layers
        }

        if epoch in unfreeze_schedule:
            stage = unfreeze_schedule[epoch]

            # Get all model layers with enhanced access
            all_layers = []
            if hasattr(yolo_internal_model, "model"):
                all_layers = list(yolo_internal_model.model.children())
            elif hasattr(yolo_internal_model, "children"):
                all_layers = list(yolo_internal_model.children())

            total_layers = len(all_layers)

            if total_layers == 0:
                print(
                    f"Warning: Could not access YOLO model layers for unfreezing at epoch {epoch}"
                )
                return

            if stage == "last_layers":
                # Unfreeze last 25% of layers
                unfreeze_from = int(total_layers * 0.75)
                for i, layer in enumerate(all_layers):
                    if i >= unfreeze_from:
                        for param in layer.parameters():
                            param.requires_grad = True
                print(
                    f"Epoch {epoch}: Unfroze last 25% of YOLO layers (from layer {unfreeze_from})"
                )

            elif stage == "middle_layers":
                # Unfreeze middle 50% of layers
                unfreeze_from = int(total_layers * 0.5)
                for i, layer in enumerate(all_layers):
                    if i >= unfreeze_from:
                        for param in layer.parameters():
                            param.requires_grad = True
                print(
                    f"Epoch {epoch}: Unfroze middle 50% of YOLO layers (from layer {unfreeze_from})"
                )

            elif stage == "early_layers":
                # Unfreeze first 75% of layers
                unfreeze_from = int(total_layers * 0.25)
                for i, layer in enumerate(all_layers):
                    if i >= unfreeze_from:
                        for param in layer.parameters():
                            param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze 75% of YOLO layers (from layer {unfreeze_from})")

            elif stage == "all_layers":
                # Unfreeze all layers
                for param in yolo_internal_model.parameters():
                    param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze all YOLO layers")

            # Reduce learning rate when unfreezing new layers
            current_lr = float(self.model.optimizer.learning_rate)
            new_lr = current_lr * 0.5
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f"Reduced learning rate from {current_lr:.2e} to {new_lr:.2e}")

    # Custom callback for progressive unfreezing - MOVED OUTSIDE the train_model method
    class ProgressiveUnfreezingCallback(tf.keras.callbacks.Callback):
        def __init__(self, model_instance):
            super().__init__()
            self.model_instance = model_instance
            self.total_epochs = None

        def on_train_begin(self, logs=None):
            self.total_epochs = self.params["epochs"]
            print(f"Progressive unfreezing enabled for {self.total_epochs} epochs")

        def on_epoch_begin(self, epoch, logs=None):
            self.model_instance.unfreeze_yolo_layers(epoch, self.total_epochs)

    def train_model(self, train_generator, val_generator, epochs=80):
        """Train model with progressive unfreezing and callbacks"""
        print("Starting YOLOv12-enhanced model training with progressive unfreezing...")

        # Setup progressive unfreezing
        yolo_internal_model = self.setup_progressive_unfreezing()

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8, min_lr=1e-8, verbose=1),
            self.ProgressiveUnfreezingCallback(self),
        ]

        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def create_confusion_matrix(self, val_generator):
        """Generate and save confusion matrix"""
        print("Generating confusion matrix...")
        save_path = os.path.join(self.save_dir, "confusion_matrix_class_weight.png")

        # Get predictions and true labels
        y_true = []
        y_pred = []

        for i in range(len(val_generator)):
            batch_x, batch_y = val_generator[i]
            predictions = self.model.predict(batch_x, verbose=0)

            y_true.extend(np.argmax(batch_y, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix - YOLOv12 Enhanced Model")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Generate classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        # Save classification report
        report_path = os.path.join(self.save_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Confusion matrix saved to: {save_path}")
        print(f"Classification report saved to: {report_path}")

    def evaluate_model_performance(self, history):
        """Analyze model performance in detail"""
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        best_val_acc = max(history.history["val_accuracy"])

        # Top-1 accuracy
        final_top1_acc = history.history.get("val_top_1_accuracy", [0])
        if final_top1_acc:
            final_top1_acc = final_top1_acc[-1]
        else:
            final_top1_acc = 0

        # Top-3 accuracy if available
        final_top3_acc = history.history.get("val_top_3_accuracy", [0])
        if final_top3_acc:
            final_top3_acc = final_top3_acc[-1]
        else:
            final_top3_acc = 0

        gap = final_train_acc - final_val_acc

        # Save performance summary
        performance_summary = {
            "final_train_accuracy": float(final_train_acc),
            "final_val_accuracy": float(final_val_acc),
            "best_val_accuracy": float(best_val_acc),
            "top1_val_accuracy": float(final_top1_acc),
            "top3_val_accuracy": float(final_top3_acc),
            "overfitting_gap": float(gap),
            "model_architecture": "YOLOv12-Enhanced EfficientNetB0",
            "total_epochs": len(history.history["accuracy"]),
        }

        summary_path = os.path.join(self.save_dir, "performance_summary.json")
        with open(summary_path, "w") as f:
            json.dump(performance_summary, f, indent=2)

        print(f"\n{'='*50}")
        print(f"YOLOV12-ENHANCED MODEL PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"Final Training Accuracy:    {final_train_acc:.4f}")
        print(f"Final Validation Accuracy:  {final_val_acc:.4f}")
        print(f"Best Validation Accuracy:   {best_val_acc:.4f}")
        print(f"Top-1 Validation Accuracy:  {final_top1_acc:.4f}")
        print(f"Top-3 Validation Accuracy:  {final_top3_acc:.4f}")
        print(f"Overfitting Gap:            {gap:.4f}")
        print(f"{'='*50}")

        if gap > 0.2:
            print("STATUS: Severe overfitting detected")
            print("RECOMMENDATION: Increase regularization or add more data")
        elif gap > 0.1:
            print("STATUS: Moderate overfitting")
            print("RECOMMENDATION: Consider additional regularization")
        else:
            print("STATUS: Good generalization")

        print(f"Performance summary saved to: {summary_path}")
        print(f"{'='*50}")

    def plot_training_history(self, history):
        """Plot comprehensive training history"""
        save_path = os.path.join(self.save_dir, "training_history.png")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("YOLOv12-Enhanced Model Training Analysis", fontsize=16, fontweight="bold")

        # Accuracy plot
        axes[0, 0].plot(history.history["accuracy"], label="Training", linewidth=2, color="blue")
        axes[0, 0].plot(
            history.history["val_accuracy"], label="Validation", linewidth=2, color="red"
        )
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss plot
        axes[0, 1].plot(history.history["loss"], label="Training", linewidth=2, color="blue")
        axes[0, 1].plot(history.history["val_loss"], label="Validation", linewidth=2, color="red")
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Top-K accuracy plot
        axes[0, 2].set_title("Top-K Accuracy")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Accuracy")

        if "val_top_1_accuracy" in history.history:
            axes[0, 2].plot(
                history.history["top_1_accuracy"],
                label="Top-1 Training",
                linewidth=2,
                color="green",
            )
            axes[0, 2].plot(
                history.history["val_top_1_accuracy"],
                label="Top-1 Validation",
                linewidth=2,
                color="darkgreen",
            )

        if "val_top_3_accuracy" in history.history:
            axes[0, 2].plot(
                history.history["top_3_accuracy"],
                label="Top-3 Training",
                linewidth=2,
                color="orange",
            )
            axes[0, 2].plot(
                history.history["val_top_3_accuracy"],
                label="Top-3 Validation",
                linewidth=2,
                color="darkorange",
            )

        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Learning rate plot
        if "lr" in history.history:
            axes[1, 0].plot(history.history["lr"], linewidth=2, color="purple")
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Overfitting analysis
        train_acc = np.array(history.history["accuracy"])
        val_acc = np.array(history.history["val_accuracy"])
        gap = train_acc - val_acc

        axes[1, 1].plot(gap, linewidth=2, color="red")
        axes[1, 1].set_title("Overfitting Gap (Train - Val)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy Difference")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.05, color="yellow", linestyle="--", alpha=0.7, label="Ideal Limit")
        axes[1, 1].axhline(
            y=0.1, color="orange", linestyle="--", alpha=0.7, label="Mild Overfitting"
        )
        axes[1, 1].axhline(
            y=0.2, color="red", linestyle="--", alpha=0.7, label="Severe Overfitting"
        )
        axes[1, 1].legend()

        # Smoothed validation accuracy
        window = min(10, len(history.history["val_accuracy"]) // 4)
        if window > 1:
            val_acc_smooth = np.convolve(
                history.history["val_accuracy"], np.ones(window) / window, mode="valid"
            )
            axes[1, 2].plot(
                range(window - 1, len(history.history["val_accuracy"])),
                val_acc_smooth,
                label=f"Val Accuracy (smoothed {window})",
                linewidth=2,
                color="red",
            )
            axes[1, 2].plot(
                history.history["val_accuracy"],
                label="Val Accuracy (original)",
                linewidth=1,
                alpha=0.5,
                color="lightcoral",
            )
            axes[1, 2].set_title("Smoothed Validation Accuracy")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("Accuracy")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training history plots saved to: {save_path}")

    def predict_with_confidence(self, image_path, threshold=0.7):
        """Make predictions with confidence analysis using YOLOv12-enhanced model"""
        if self.model is None:
            return None

        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return None

        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image, verbose=0)

        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [(i, predictions[0][i]) for i in top_5_indices]

        result = {
            "predicted_class": (
                self.class_names[predicted_class]
                if predicted_class < len(self.class_names)
                else f"Class_{predicted_class}"
            ),
            "confidence": float(confidence),
            "class_id": int(predicted_class),
            "is_confident": confidence > threshold,
            "top_5_predictions": [
                {
                    "class": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                    "confidence": float(conf),
                    "class_id": int(i),
                }
                for i, conf in top_5_predictions
            ],
            "entropy": float(-np.sum(predictions[0] * np.log(predictions[0] + 1e-8))),
            "model_type": "YOLOv12-Enhanced",
        }

        return result

    def save_model(self):
        """Save the final trained model with metadata in the specific folder"""
        if self.model:
            model_path = os.path.join(self.save_dir, "furniture_model_yolov12_final.h5")
            self.model.save(model_path)

            # Save metadata
            metadata = {
                "class_names": self.class_names,
                "img_size": self.img_size,
                "num_classes": len(self.class_names),
                "model_type": "YOLOv12-Enhanced EfficientNetB0",
                "yolo_model_used": self.yolo_model is not None,
                "training_timestamp": datetime.datetime.now().isoformat(),
                "save_directory": self.save_dir,
            }

            metadata_path = os.path.join(self.save_dir, "model_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save model architecture summary
            with open(os.path.join(self.save_dir, "model_summary.txt"), "w") as f:
                self.model.summary(print_fn=lambda x: f.write(x + "\n"))

            print(f"Final model saved to: {model_path}")
            print(f"Metadata saved to: {metadata_path}")
            print(f"Model summary saved to: {os.path.join(self.save_dir, 'model_summary.txt')}")

    def load_model(self, filepath="Models/furniture_model_final.h5"):
        """Load a previously trained model"""
        self.model = tf.keras.models.load_model(filepath)

        # Load metadata
        metadata_path = filepath.replace(".h5", "_metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.class_names = metadata.get("class_names", [])
                self.img_size = tuple(metadata.get("img_size", (640, 640)))
            print(f"Model and metadata loaded from: {filepath}")
        except FileNotFoundError:
            print("Metadata file not found, using default values")


def main():
    """Main training pipeline with YOLOv12 integration"""
    # Initialize model
    furniture_model = ImprovedFurnitureModelYOLOv12(img_size=(640, 640), batch_size=32)

    # Initialize YOLOv12
    furniture_model.initialize_yolov12(model_size="n")

    # Download and prepare dataset
    gdrive_file_id = "1Mfp9TV22_2eU47nZYU00LAs8HkPwAqYZ"
    dataset_path = furniture_model.download_and_extract_remote_dataset(gdrive_file_id)

    # Parse annotations with 80/20 split
    train_data, val_data = furniture_model.parse_yolo_annotations_with_split(
        dataset_path, train_split=0.8
    )

    print(f"Dataset prepared: {len(train_data[0])} training, {len(val_data[0])} validation samples")

    # Create data generators
    train_generator, val_generator = furniture_model.create_data_generators(train_data, val_data)

    # Build model
    model = furniture_model.build_model(num_classes=len(furniture_model.class_names))
    print(f"Model parameters: {model.count_params():,}")
    print(f"Number of classes: {len(furniture_model.class_names)}")

    # Train model
    history = furniture_model.train_model(train_generator, val_generator, epochs=80)

    # Generate and save confusion matrix
    furniture_model.create_confusion_matrix(val_generator)

    # Analyze and save visualizations
    furniture_model.plot_training_history(history)
    furniture_model.evaluate_model_performance(history)

    # Save final model (now uses the instance's save_dir)
    furniture_model.save_model()

    # Clean up temporary dataset
    if os.path.exists("dataset/"):
        shutil.rmtree("dataset/")
        print("Temporary dataset cleaned up")

    print("Training pipeline completed successfully!")
    print(f"All outputs saved to: {furniture_model.save_dir}")
    print("Check the folder for:")
    print("- Final model: furniture_model_yolov12_final.h5")
    print("- Training plots: training_history.png")
    print("- Confusion matrix: confusion_matrix.png")
    print("- Classification report: classification_report.json")
    print("- Performance summary: performance_summary.json")


if __name__ == "__main__":
    main()
