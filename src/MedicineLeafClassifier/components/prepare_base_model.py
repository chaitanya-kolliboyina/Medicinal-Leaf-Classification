import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from MedicineLeafClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBasemodel:
    def __init__(self, config:PrepareBaseModelConfig):
        self.config = config


    def get_base_model(self):
        self.model = tf.keras.applications.InceptionResNetV2(
            input_shape=self.config.params_image_size,
            weights = self.config.params_weight,
            include_top = self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)     


    @staticmethod
    def prepare_full_model(model,classes,freeze,boundaries,learning_rate):
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries,learning_rate
        )
        
        if freeze:
            model.trainable = False

        full_model = tf.keras.Sequential([
            tf.keras.layers.Input((224,224,3)),
            tf.keras.layers.Rescaling(1./255),
            model, 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(classes, activation='softmax')  # Adjust for your number of classes
            ])
        
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']
        )
        full_model.summary()

        return full_model
    
    def update_base_model(self):
        self.full_model = self.prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze=self.config.params_freeze,
            boundaries=self.config.params_boundaries,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_model_path, model=self.full_model)

    @staticmethod
    def save_model(path:Path, model):
        model.save(path)