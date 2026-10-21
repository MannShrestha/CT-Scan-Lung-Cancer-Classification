import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.cnnClassifier.entity.config_entity import EvaluationConfig
from src.cnnClassifier.utils.common import read_yaml, create_directories,save_json

import dagshub
dagshub.init(repo_owner='MannShrestha', repo_name='CT-Scan-Lung-Cancer-Classification', mlflow=True)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = {}

    def _valid_generator(self):
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

        self.valid_generator = (
            valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(
            self.config.path_of_model
        )

        self._valid_generator()

        # Evaluate the model and return named metrics
        self.score = self.model.evaluate(
            self.valid_generator,
            return_dict=True,
            verbose=1
        )

        self.save_score()

        return self.score

    def save_score(self):
        scores = {
            metric: float(value)
            for metric, value in self.score.items()
        }

        save_json(
            path=Path("scores.json"),
            data=scores
        )

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)

        with mlflow.start_run():
            # MLflow parameters must be serializable as strings
            params = {
                key: str(value)
                for key, value in self.config.all_params.items()
            }

            mlflow.log_params(params)

            # Log all evaluation metrics by name
            mlflow.log_metrics({
                metric: float(value)
                for metric, value in self.score.items()
            })

            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()
            ).scheme

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    name="model",
                    registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(
                    self.model,
                    name="model"
                )
