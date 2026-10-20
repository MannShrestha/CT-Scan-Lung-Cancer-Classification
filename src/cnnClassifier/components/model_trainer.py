import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Load the model without reusing its previous optimizer state
        self.model = tf.keras.models.load_model(
        self.config.updated_base_model_path, compile=False)

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = (
            tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = (
                tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **datagenerator_kwargs
                )
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.params_learning_rate
        )

        metric_map = {
            "accuracy": tf.keras.metrics.CategoricalAccuracy(
                name="accuracy"
            ),
            "precision": tf.keras.metrics.Precision(
                name="precision"
            ),
            "recall": tf.keras.metrics.Recall(
                name="recall"
            )
        }

        metrics = []

        for metric_name in self.config.params_metrics:
            key = metric_name.lower()

            if key not in metric_map:
                raise ValueError(
                    f"Unsupported metric: {metric_name}"
                )

            metrics.append(metric_map[key])

        self.model.compile(
            optimizer=optimizer,
            loss=self.config.params_loss,
            metrics=metrics
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        # len(generator) includes the final partial batch
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
