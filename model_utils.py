import tensorflow as tf
import urllib.request


def load_model_from_github(url):
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    custom_objects = {'TransformerEncoderBlock': TransformerEncoderBlock}
    loaded_model = tf.keras.models.load_model(filename, custom_objects=custom_objects)
    return loaded_model
