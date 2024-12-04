import tensorflow as tf

def load_model(model_path):
    """Loads the pre-trained model from the given path."""
    model = tf.keras.models.load_model(model_path)
    return model
