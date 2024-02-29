from MedicineLeafClassifier.entity.config_entity import TrainingConfig  
import tensorflow as tf
from pathlib import Path 


class Training:
    def __init__(self,config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_split(self):
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
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
             subset='training',
             )
        
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

        train_data = train_data.cache().prefetch(buffer_size = AUTOTUNE)

        valid_data = valid_data.cache().prefetch(buffer_size = AUTOTUNE)

        return train_data, valid_data
    

    @staticmethod
    def save_model(path: Path, model = tf.keras.Model):
        """ Saves the model after last epoch
          irrespective of best accuracy or not """
        model.save(path)
    
    
    def train(self, train_data, valid_data, callbacks_list: list):
        self.model.fit(
            train_data,
            epochs = self.config.params_epochs,
            
            validation_data = valid_data,
            
            callbacks = callbacks_list
        )

        self.save_model(
            path = self.config.trained_model_path,
             model = self.model )