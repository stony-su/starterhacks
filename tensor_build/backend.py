from flask import Flask, render_template
import tensorflow
from tensorflow import keras
from keras import layers

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

'''
backend logic:
-> dragable blocks in html sends string of what blocks are connceted/snapped together
-> this string will represent all functions we have to run
-> functions splti using "match function"
-> functions run 
-> scrap tensorbuild.blocks functions?
-> we can hard code (fix the variables) for the functions
'''


"""
javascirpt logic:
-> sends string from html to flask

fetch("http://localhost:5000/compile_model", (data) => {
    
})

@app.route('/compile_model')

"""

def matchFunction(comsep: str):
    comseplist = comsep.split(",")
    # layer1,layer2,compile,fit
    for i in comseplist:
        if "layer1":
            layer1(model)
        elif "layer2"

def layer1(model):
    """
    Adds a 2D convolutional layer to the given TensorFlow CNN model with hard-coded parameters.

    Args:
        model (tf.keras.Model): The model to which the Conv2D layer will be added.

    Returns:
        tf.keras.Model: The updated model with the Conv2D layer added.
    """
    # Define the hard-coded parameters for the Conv2D layer
    filters = 32
    kernel_size = (3, 3)
    activation = 'relu'
    padding = 'same'
    strides = (1, 1)
    input_shape = (64, 64, 3)  # Example input shape, adjust as necessary

    # Add the Conv2D layer
    if len(model.layers) == 0:
        # If the model has no layers, add the Conv2D layer as the first layer
        model.add(layers.Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                activation=activation,
                                padding=padding,
                                strides=strides,
                                input_shape=input_shape))
    else:
        # If the model already has layers, add the Conv2D layer after the existing layers
        model.add(layers.Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                activation=activation,
                                padding=padding,
                                strides=strides))
    
    return model


if __name__ == '__main__':
    app.run(debug=True)
