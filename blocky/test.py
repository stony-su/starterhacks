import tensorflow as tf
from blocks.dense_block import DenseBlock
from blocks.activation_block import ActivationBlock

def test_dense_block():
    dense_block = DenseBlock(units=32, activation='relu')
    inputs = tf.keras.Input(shape=(16,))
    outputs = dense_block.apply(inputs)
    
    assert outputs.shape[-1] == 32
    print("DenseBlock test passed.")

def test_activation_block():
    activation_block = ActivationBlock(activation='softmax')
    inputs = tf.keras.Input(shape=(32,))
    outputs = activation_block.apply(inputs)
    
    assert outputs.shape[-1] == 32
    print("ActivationBlock test passed.")

if __name__ == "__main__":
    test_dense_block()
    test_activation_block()