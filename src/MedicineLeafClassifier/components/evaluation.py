import tensorflow as tf
from pathlib import Path
from MedicineLeafClassifier.entity.config_entity import EvaluationConfig
from MedicineLeafClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def valid_data(self):
        valid_data = tf.keras.preprocessing.image_dataset_from_directory(
             self.config.training_data,
             labels = 'inferred',
             label_mode = 'int',
             color_mode = 'rgb',
            class_names = None,
             batch_size = self.config.params_batch_size,
             image_size= (self.config.params_image_size[0], self.config.params_image_size[1]),
             shuffle=True,
             seed=self.config.params_seed,
             validation_split=0.1,
             subset='validation',
             )
        AUTOTUNE = tf.data.AUTOTUNE

        # train_data = train_data.cache().prefetch(buffer_size = AUTOTUNE)

        valid_data = valid_data.cache().prefetch(buffer_size = AUTOTUNE)

        # return valid_data
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self.valid_data()
        self.score = model.evaluate(self.valid_data)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
