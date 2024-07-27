import tensorflow as tf
class Block:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    def to_code(self):
        # Convert block to TensorFlow code
        pass

class DenseBlock:
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.layer = tf.keras.layers.Dense(units, activation=activation)

    def apply(self, inputs):
        return self.layer(inputs)

    def get_config(self):
        return {
            'units': self.units,
            'activation': self.activation
        }


class ActivationBlock:
    def __init__(self, activation):
        self.activation = activation
        self.layer = tf.keras.layers.Activation(activation)

    def apply(self, inputs):
        return self.layer(inputs)

    def get_config(self):
        return {
            'activation': self.activation
        }
