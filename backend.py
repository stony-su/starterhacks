from flask import Flask, render_template
import tensorflow
from tensorflow import keras
from keras import callbacks
from flask_cors import CORS
from keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import datasets
from flask import request, jsonify

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/blocks')
def blocks():
    return render_template('blocks.html')


model = models.Sequential()

@app.route('/compile_model', methods=['POST'])
def matchFunction():
    print('hello world')
    data = request.json
    comsep = data.get('code', '')
    comseplist = comsep.split(",")
    # layer1,layer2,compile,fit
    for i in comseplist:
        if "conv2d" in i:
            conv_layer(model)
        elif "dense_layer":
            dense_layer(model)
        elif "drop_layer":
            dropout_layer(model)
        elif "max_layer":
            maxpooling_layer(model)
        elif "active_layer":
            activation_layer(model)
        elif "average_layer":
            averagepooling_layer(model)
        elif "flat_layer":
            flatten_layer(model)
        elif "output_layer":
            output_layer(model)
        elif "build_layer":
            build_model(model)
        elif "summary_layer":
            run_model(model)
        elif "train_layer":
            fit_model(model)
        elif "preprocess_layer":
            preprocess_and_build_model(model)
    return jsonify({'status': 'success'})

@app.route('/preprocess_layer')
def preprocess_and_build_model(test_size=0.2, random_state=42):
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # Flatten the images and normalize the pixel values
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val

@app.route('/conv_layer')
def conv_layer(model):
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

@app.route('/dense_layer')
def dense_layer(model):
    # Define fixed parameters for the Dense layer
    units = 10
    activation = 'softmax'

    # Add the Dense layer with fixed parameters
    model.add(layers.Dense(units=units, activation=activation))

    return model

@app.route('/drop_layer')
def dropout_layer(model, rate=0.5):
    model.add(layers.Dropout(rate=rate))
    return model

@app.route('/max_layer')
def maxpooling_layer(model, pool_size=(2, 2), strides=(2, 2), padding='valid'):
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))
    return model

@app.route('/active_layer')
def activation_layer(model, activation='relu'):
    model.add(layers.Activation(activation=activation))
    return model

@app.route('/average_layer')
def averagepooling_layer(model, pool_size=(2, 2), strides=(2, 2), padding='valid'):
    model.add(layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding))
    return model

@app.route('/flat_layer')
def flatten_layer(model):
    model.add(layers.Flatten())
    return model

@app.route('/output_layer')
def output_layer(model, units=10, activation='softmax'):
    model.add(layers.Dense(units=units, activation=activation))
    return model


@app.route('/build_layer')
def build_model():
    # Create a new Sequential model
    model = models.Sequential()

    # Add layers to the model
    model = conv_layer(model)
    model = maxpooling_layer(model)
    model = activation_layer(model)
    model = dropout_layer(model)
    model = averagepooling_layer(model)
    model = flatten_layer(model)
    model = output_layer(model)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

@app.route('/summary_layer')
def run_model(model):
    model = build_model()
    model.summary()
    return model.summary

@app.route('/train_layer')
def fit_model(model, train_data, val_data=None, epochs=10, batch_size=32, patience=3):
    # Define early stopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Check if the data is in the form of (X, y) tuples
    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        if val_data is not None:
            if isinstance(val_data, tuple):
                X_val, y_val = val_data
                history = model.fit(X_train, y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(X_val, y_val),
                                    callbacks=[early_stopping])
            else:
                history = model.fit(X_train, y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=val_data,
                                    callbacks=[early_stopping])
        else:
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping])
    else:
        if val_data is not None:
            history = model.fit(train_data,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=val_data,
                                callbacks=[early_stopping])
        else:
            history = model.fit(train_data,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping])

    # Print results
    print("Training completed.")
    print(f"Final training loss: {history.history['loss'][-1]}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]}")
    
    if 'val_loss' in history.history:
        print(f"Final validation loss: {history.history['val_loss'][-1]}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]}")

    return history

if __name__ == '__main__':
    app.run(debug=True, port = 5000)
