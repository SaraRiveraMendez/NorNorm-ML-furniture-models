import json
import os
import zipfile
from pathlib import Path

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


class ImprovedFurnitureModel:
    def __init__(
        self, img_size=(224, 224), batch_size=16
    ):  # Tamaño más estándar y batch size mayor
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_names = []

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

    def parse_yolo_annotations(self, dataset_path):
        """Extract ALL elements from each image, creating multiple training samples per image"""
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get("names", [])
                print(f"Classes found: {self.class_names}")

        train_images = []
        train_labels = []
        val_images = []
        val_labels = []

        total_elements = 0
        images_processed = 0

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
                                    if len(parts) >= 5:  # class_id, x, y, w, h
                                        class_id = int(parts[0])

                                        # Verify valid class_id
                                        if 0 <= class_id < len(self.class_names):
                                            if split == "train":
                                                train_images.append(img_path)
                                                train_labels.append(class_id)
                                            else:
                                                val_images.append(img_path)
                                                val_labels.append(class_id)

                                            elements_in_image += 1
                                            total_elements += 1

                            if elements_in_image > 0:
                                images_processed += 1

        print(f"Images processed: {images_processed}")
        print(f"Total elements extracted: {total_elements}")
        print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
        print(f"Average elements per image: {total_elements/images_processed:.2f}")

        # Show class distribution
        print(f"Class distribution in training:")
        unique, counts = np.unique(train_labels, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"Unknown_{class_id}"
            )
            print(f"  {class_name}: {count} samples")

        return (train_images, train_labels), (val_images, val_labels)

    def load_and_preprocess_image(self, image_path):
        """Preprocesamiento mejorado de imágenes"""
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

    def create_data_generators_with_augmentation(self, train_data, val_data):
        """Generadores de datos con data augmentation"""
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Convertir labels a categórico
        num_classes = len(self.class_names)
        y_train_cat = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
        y_val_cat = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)

        # Data augmentation para entrenamiento
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode="nearest",
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
        )

        # Sin augmentation para validación
        val_datagen = ImageDataGenerator()

        class ImprovedSequence(tf.keras.utils.Sequence):
            def __init__(
                self, image_paths, labels, batch_size, img_size, datagen=None, shuffle=True
            ):
                self.image_paths = image_paths
                self.labels = labels
                self.batch_size = batch_size
                self.img_size = img_size
                self.datagen = datagen
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
                            # Aplicar data augmentation si está disponible
                            if self.datagen is not None:
                                image = image.reshape((1,) + image.shape)
                                image = self.datagen.flow(image, batch_size=1)[0][0]

                            batch_images.append(image)
                            batch_labels.append(label)

                # Asegurar que tenemos al menos una imagen
                if not batch_images:
                    # Crear batch vacío válido
                    batch_images = [np.zeros((*self.img_size, 3))]
                    batch_labels = [
                        np.zeros(len(self.labels[0]) if len(self.labels) > 0 else num_classes)
                    ]

                return np.array(batch_images), np.array(batch_labels)

            def load_and_preprocess_image_batch(self, image_path):
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
            train_images,
            y_train_cat,
            self.batch_size,
            self.img_size,
            datagen=train_datagen,
            shuffle=True,
        )
        val_generator = ImprovedSequence(
            val_images, y_val_cat, self.batch_size, self.img_size, datagen=None, shuffle=False
        )

        return train_generator, val_generator

    def build_improved_model(self, num_classes):
        """Arquitectura mejorada y más balanceada"""
        # Usar EfficientNetB0 que es más moderno y eficiente
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*self.img_size, 3), include_top=False, weights="imagenet"
        )

        # Estrategia de fine-tuning más inteligente
        base_model.trainable = True

        # Congelar las primeras capas, descongelar las últimas
        fine_tune_at = len(base_model.layers) - 30
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Arquitectura más balanceada
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),  # Dropout más moderado
                layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Learning rate más alto inicialmente
        initial_learning_rate = 1e-4

        # Crear métrica top-3 personalizada si no existe
        try:
            # Intentar usar la métrica estándar
            top3_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
            metrics = ["accuracy", top3_metric]
        except AttributeError:
            # Si no existe, usar solo accuracy
            metrics = ["accuracy"]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        self.model = model
        return model

    def train_with_progressive_learning(self, train_generator, val_generator, epochs=100):
        """Entrenamiento progresivo con diferentes fases"""

        # Fase 1: Entrenamiento con capas congeladas (warm-up)
        print("=== FASE 1: Warm-up (capas base congeladas) ===")

        # Congelar todas las capas del modelo base
        for layer in self.model.layers[0].layers:
            layer.trainable = False

        # Recompilar con learning rate más alto
        try:
            top3_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
            metrics = ["accuracy", top3_metric]
        except AttributeError:
            metrics = ["accuracy"]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        callbacks_phase1 = [
            EarlyStopping(
                monitor="val_accuracy", patience=10, restore_best_weights=True, min_delta=0.001
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ]

        # Entrenar primera fase
        history1 = self.model.fit(
            train_generator,
            epochs=min(30, epochs // 3),
            validation_data=val_generator,
            callbacks=callbacks_phase1,
            verbose=1,
        )

        # Fase 2: Fine-tuning completo
        print("\n=== FASE 2: Fine-tuning completo ===")

        # Descongelar capas gradualment
        for layer in self.model.layers[0].layers[-50:]:  # Últimas 50 capas
            layer.trainable = True

        # Recompile with lower learning rate
        try:
            top3_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
            metrics = ["accuracy", top3_metric]
        except AttributeError:
            metrics = ["accuracy"]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        callbacks_phase2 = [
            EarlyStopping(
                monitor="val_accuracy", patience=20, restore_best_weights=True, min_delta=0.0005
            ),
            ModelCheckpoint(
                "Models/best_furniture_model.h5",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8, min_lr=1e-8, verbose=1),
        ]

        # Entrenar segunda fase
        remaining_epochs = epochs - len(history1.history["loss"])
        history2 = self.model.fit(
            train_generator,
            epochs=remaining_epochs,
            validation_data=val_generator,
            callbacks=callbacks_phase2,
            verbose=1,
        )

        # Combinar historiales
        combined_history = self.combine_histories(history1, history2)

        return combined_history

    def combine_histories(self, hist1, hist2):
        """Combinar dos historiales de entrenamiento"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]

        # Crear objeto tipo History
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict

        return CombinedHistory(combined)

    def evaluate_model_performance(self, history):
        """Análisis detallado del rendimiento"""
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        best_val_acc = max(history.history["val_accuracy"])

        # Top-3 accuracy si está disponible
        final_top3_acc = history.history.get("val_top_3_accuracy", [0])[-1]

        gap = final_train_acc - final_val_acc

        print(f"\n{'='*50}")
        print(f"RESUMEN DE RENDIMIENTO DEL MODELO")
        print(f"{'='*50}")
        print(f"Accuracy Final Entrenamiento: {final_train_acc:.4f}")
        print(f"Accuracy Final Validación:    {final_val_acc:.4f}")
        print(f"Mejor Accuracy Validación:    {best_val_acc:.4f}")
        print(f"Top-3 Accuracy Validación:    {final_top3_acc:.4f}")
        print(f"Gap Overfitting:              {gap:.4f}")
        print(f"{'='*50}")

        if gap > 0.2:
            print("❌ ESTADO: Overfitting severo detectado")
            print("💡 RECOMENDACIÓN: Aumentar regularización o más datos")
        elif gap > 0.1:
            print("⚠️  ESTADO: Overfitting moderado")
            print("💡 RECOMENDACIÓN: Considerar más regularización")
        else:
            print("✅ ESTADO: Buena generalización")

        print(f"{'='*50}")

    def plot_comprehensive_training_history(self, history):
        """Visualización completa del entrenamiento"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Análisis Completo del Entrenamiento", fontsize=16, fontweight="bold")

        # Accuracy
        axes[0, 0].plot(
            history.history["accuracy"], label="Entrenamiento", linewidth=2, color="blue"
        )
        axes[0, 0].plot(
            history.history["val_accuracy"], label="Validación", linewidth=2, color="red"
        )
        axes[0, 0].set_title("Accuracy del Modelo")
        axes[0, 0].set_xlabel("Época")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(history.history["loss"], label="Entrenamiento", linewidth=2, color="blue")
        axes[0, 1].plot(history.history["val_loss"], label="Validación", linewidth=2, color="red")
        axes[0, 1].set_title("Loss del Modelo")
        axes[0, 1].set_xlabel("Época")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Top-3 Accuracy (si está disponible)
        if "val_top_3_accuracy" in history.history:
            axes[0, 2].plot(
                history.history["top_3_accuracy"],
                label="Top-3 Entrenamiento",
                linewidth=2,
                color="green",
            )
            axes[0, 2].plot(
                history.history["val_top_3_accuracy"],
                label="Top-3 Validación",
                linewidth=2,
                color="orange",
            )
            axes[0, 2].set_title("Top-3 Accuracy")
            axes[0, 2].set_xlabel("Época")
            axes[0, 2].set_ylabel("Top-3 Accuracy")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # Learning Rate
        if "lr" in history.history:
            axes[1, 0].plot(history.history["lr"], linewidth=2, color="purple")
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Época")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Análisis de Overfitting
        train_acc = np.array(history.history["accuracy"])
        val_acc = np.array(history.history["val_accuracy"])
        gap = train_acc - val_acc

        axes[1, 1].plot(gap, linewidth=2, color="red")
        axes[1, 1].set_title("Gap de Overfitting (Train - Val)")
        axes[1, 1].set_xlabel("Época")
        axes[1, 1].set_ylabel("Diferencia de Accuracy")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.05, color="yellow", linestyle="--", alpha=0.7, label="Límite Ideal")
        axes[1, 1].axhline(
            y=0.1, color="orange", linestyle="--", alpha=0.7, label="Overfitting Leve"
        )
        axes[1, 1].axhline(
            y=0.2, color="red", linestyle="--", alpha=0.7, label="Overfitting Severo"
        )
        axes[1, 1].legend()

        # Suavizado de métricas
        window = min(10, len(history.history["val_accuracy"]) // 4)
        if window > 1:
            val_acc_smooth = np.convolve(
                history.history["val_accuracy"], np.ones(window) / window, mode="valid"
            )
            axes[1, 2].plot(
                range(window - 1, len(history.history["val_accuracy"])),
                val_acc_smooth,
                label=f"Val Accuracy (suavizado {window})",
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
            axes[1, 2].set_title("Accuracy Validación Suavizada")
            axes[1, 2].set_xlabel("Época")
            axes[1, 2].set_ylabel("Accuracy")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_with_confidence_analysis(self, image_path, threshold=0.7):
        """Predicción mejorada con análisis de confianza"""
        if self.model is None:
            return None

        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return None

        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image, verbose=0)

        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Top 5 predicciones
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
            "entropy": float(
                -np.sum(predictions[0] * np.log(predictions[0] + 1e-8))
            ),  # Medida de incertidumbre
        }

        return result

    def save_model(self, filepath="Models/improved_furniture_model.h5"):
        if self.model:
            os.makedirs("Models", exist_ok=True)
            self.model.save(filepath)

            # Guardar metadatos
            metadata = {
                "class_names": self.class_names,
                "img_size": self.img_size,
                "num_classes": len(self.class_names),
                "model_type": "EfficientNetB0",
            }

            with open(filepath.replace(".h5", "_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Modelo y metadatos guardados en: {filepath}")

    def load_model(self, filepath="Models/improved_furniture_model.h5"):
        self.model = tf.keras.models.load_model(filepath)

        # Cargar metadatos
        metadata_path = filepath.replace(".h5", "_metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.class_names = metadata.get("class_names", [])
                self.img_size = tuple(metadata.get("img_size", (224, 224)))
            print(f"Modelo y metadatos cargados desde: {filepath}")
        except FileNotFoundError:
            print("Archivo de metadatos no encontrado, usando valores por defecto")


# Función principal mejorada
def main():
    print("🚀 Iniciando entrenamiento del modelo mejorado de muebles")

    # Configuración optimizada
    furniture_model = ImprovedFurnitureModel(
        img_size=(224, 224),  # Tamaño estándar para EfficientNet
        batch_size=16,  # Batch size más grande
    )

    # Descargar y procesar dataset
    gdrive_file_id = "1i3cNtxQ0xZTn2-ytDYMLpmYEgBGSI3UP"
    dataset_path = furniture_model.download_and_extract_remote_dataset(gdrive_file_id)
    train_data, val_data = furniture_model.parse_yolo_annotations(dataset_path)

    # Verificar que tenemos datos suficientes
    if len(train_data[0]) == 0:
        print("❌ Error: No se encontraron datos de entrenamiento")
        return

    print(f"✅ Datos cargados: {len(train_data[0])} entrenamiento, {len(val_data[0])} validación")

    # Crear generadores con data augmentation
    train_generator, val_generator = furniture_model.create_data_generators_with_augmentation(
        train_data, val_data
    )

    # Construir modelo mejorado
    model = furniture_model.build_improved_model(num_classes=len(furniture_model.class_names))
    print(f"📊 Parámetros del modelo: {model.count_params():,}")
    print(f"📊 Número de clases: {len(furniture_model.class_names)}")

    # Entrenamiento progresivo
    history = furniture_model.train_with_progressive_learning(
        train_generator, val_generator, epochs=80
    )

    # Análisis y visualización
    furniture_model.plot_comprehensive_training_history(history)
    furniture_model.evaluate_model_performance(history)

    # Guardar modelo
    furniture_model.save_model("Models/improved_furniture_model.h5")

    # Limpiar dataset para ahorrar espacio
    import shutil

    if os.path.exists("dataset/"):
        shutil.rmtree("dataset/")
        print("🧹 Dataset temporal limpiado")

    print("🎉 Entrenamiento completado!")


if __name__ == "__main__":
    main()
