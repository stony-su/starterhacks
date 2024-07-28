import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras import datasets
from tensor_build.blocks import Conv2DBlock, DenseBlock, DropoutBlock, PoolingBlock

def load_data():
    """
    Load and preprocess CIFAR-10 dataset.
    Returns:
        Tuple of numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """
    Create a CNN model using custom blocks.
    
    Returns:
        A compiled tf.keras.Model instance.
    """
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    # Create instances of your custom blocks
    conv_block1 = Conv2DBlock(filters=32, kernel_size=(3, 3), activation='relu')
    conv_block2 = Conv2DBlock(filters=64, kernel_size=(3, 3), activation='relu')
    pool_block = PoolingBlock(pool_size=(2, 2))
    dropout_block = DropoutBlock(rate=0.5)
    dense_block1 = DenseBlock(units=128, activation='relu')
    dense_block2 = DenseBlock(units=10, activation='softmax')
    
    # Apply blocks  
    x = conv_block1.apply(inputs)
    x = conv_block2.apply(x)
    x = pool_block.apply(x)
    x = tf.keras.layers.Flatten()(x)
    x = dropout_block.apply(x, training=True)
    x = dense_block1.apply(x)
    outputs = dense_block2.apply(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model():
    """
    Load data, create and train the model.
    
    Returns:
        The history object from model training.
    """
    (x_train, y_train), (x_test, y_test) = load_data()
    
    model = create_model()
    
    # Model summary
    model.summary()
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
    
    return history
