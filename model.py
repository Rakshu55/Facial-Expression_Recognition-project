# Let us import the Libraries required.
import numpy as np
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# To use the model saved in the Json format, We are importing "model_from_json"
from tensorflow.keras.models import model_from_json


class FacialExpressionModel(object):

    """ A Class for Predicting the emotions using the pre-trained Model weights"""

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    # Whenever we create an instance of class , these are initialized
    def __init__(self, model_json_file, model_weights_file):

        # Now Let us load model from JSON file which we created during Training
        with open(model_json_file, "r") as json_file:

            # Reading the json file and storing it in loaded_model
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # Now, Let us load weights into the model
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        """ It predicts the Emotion using our pre-trained model and returns it """

        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

    def return_probabs(self, img):
        """  It returns the Probabilities of each emotions using pre-trained model """

        self.preds = self.loaded_model.predict(img)
        return self.preds
