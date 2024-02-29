import os 
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from MedicineLeafClassifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self,config:PrepareCallbacksConfig):
        self.config = config

    
    @property
    def create_tb_callback(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir = tb_running_log_dir)
    @property
    def create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_model_filepath,"model_{epoch}.h5"),  # Replace with your desired path
            save_best_only=True,  # Set to True to save only the best model based on a metric
            monitor='val_accuracy',  # Monitor validation accuracy during training
            save_weights_only=False,  # Set to True to save only model weights
            verbose=1  # Set to 0 for silent operation
        )
    @property
    def create_early_stopping(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Monitor validation accuracy
            min_delta=0.01,  # Minimum required change in the monitored metric
            patience=50,  # Number of epochs with no improvement to wait before stopping
            baseline = 0.5,
            restore_best_weights=True  # Restore the weights of the best model before stopping
        )
    @property
    def get_tb_ckpt_callbacks(self):
        return [
            self.create_tb_callback,
            self.create_ckpt_callbacks,
            self.create_early_stopping
        ]