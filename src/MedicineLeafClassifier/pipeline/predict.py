import numpy as np
import tensorflow as tf 
# from tensorflow.keras.saving import load_model  
# from tensorflow.keras.utils import image
import os

classnames = os.listdir(os.path.join("artifacts","data_ingestion","indian-medicinal-leaf-image-dataset","Medicinal Leaf dataset"))

class PredictPipeline:
    def __init__(self,filename):
        self.filename = filename


    def predict(self):
        #load_model 
        model = tf.keras.saving.load_model(os.path.join("artifacts","training","model.h5"))


        imagename = self.filename
        test_image = tf.keras.utils.image.load_img(imagename,target_size = (224,224))
        test_image = tf.keras.utils.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image,verbose = 0)
        predicted_class = classnames[np.argmax(result[0])]
        confidence = round(100 * (np.max(result[0])),2)
        # result = np.argmax(model.predict(test_image),axis = 1)
        

        return predicted_class,confidence

        

