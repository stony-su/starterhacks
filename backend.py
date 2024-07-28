from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from flask_cors import CORS
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)
CORS(app)


# def generate_accuracy_plot(model_number, save_dir='plots'):
#     epochs = np.arange(1, 21)
#     accuracy = np.random.rand(20).cumsum()
#     accuracy /= accuracy.max()  # Normalize to range [0, 1]

#     plt.figure()
#     plt.plot(epochs, accuracy, marker='o', label=f'Model {model_number}')
#     plt.title(f'Accuracy Rate of Model {model_number}')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.grid(True)

#     # Ensure the directory exists
#     os.makedirs(save_dir, exist_ok=True)

#     # Save the plot
#     plot_path = os.path.join(save_dir, f'model_{model_number}.png')
#     plt.savefig(plot_path)
#     plt.close()

#     return plot_path

def preprocess_and_build_model(test_size=0.2, random_state=42):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val

def conv_layer(model):
    filters = 32
    kernel_size = (3, 3)
    activation = 'relu'
    padding = 'same'
    strides = (1, 1)
    input_shape = (28, 28, 1)
    if len(model.layers) == 0:
        model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, strides=strides, input_shape=input_shape))
    else:
        model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, strides=strides))
    return model

def dense_layer(model):
    units = 10
    activation = 'softmax'
    model.add(layers.Dense(units=units, activation=activation))
    return model

def dropout_layer(model, rate=0.5):
    model.add(layers.Dropout(rate=rate))
    return model

def maxpooling_layer(model, pool_size=(2, 2), strides=(2, 2), padding='valid'):
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))
    return model

def activation_layer(model, activation='relu'):
    model.add(layers.Activation(activation=activation))
    return model

def averagepooling_layer(model, pool_size=(2, 2), strides=(2, 2), padding='valid'):
    model.add(layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding))
    return model

def flatten_layer(model):
    model.add(layers.Flatten())
    return model

def output_layer(model, units=10, activation='softmax'):
    model.add(layers.Dense(units=units, activation=activation))
    return model

def build_model():
    model = models.Sequential()
    model = conv_layer(model)
    model = maxpooling_layer(model)
    model = activation_layer(model)
    model = dropout_layer(model)
    model = averagepooling_layer(model)
    model = flatten_layer(model)
    model = output_layer(model)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_model():
    model = build_model()
    model.summary()
    return model.summary()

def fit_model(model, file_path='/Users/eddieb/Coding Projects/starterhacks/models'
              #X_train, y_train, X_val, y_val, epochs=10, batch_size=32, patience=3
    ):

    """
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
    history_dict = {
        "loss": history.history.get("loss", []),
        "accuracy": history.history.get("accuracy", []),
        "val_loss": history.history.get("val_loss", []),
        "val_accuracy": history.history.get("val_accuracy", [])
    }
    with open('history.json', 'w') as f:
        json.dump(history_dict, f)
    print("BREH")"""
    if file_path.endswith('.h5'):
        model.save(file_path)
    else:
        model.save(file_path)
    
    return file_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/blocks')
def blocks():
    # Directory containing your images
    image_dir = os.path.join(app.static_folder, 'images/')
    
    # List all image filenames in the directory
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    # Generate full URLs for the images
    image_urls = [url_for('static', filename=f'images/{image}') for image in image_filenames]
    
    return render_template('blocks.html', image_urls=image_urls)

@app.route('/compile_model', methods=['POST'])
def matchFunction():
    model = models.Sequential()
    data = request.json
    comsep = data.get('code', '')
    comseplist = comsep.split(",")[:-1]
    X_train, X_val, y_train, y_val = preprocess_and_build_model()
    for i in range(1, 11):
        # generate_accuracy_plot(i)
        print("")
    """for i in comseplist:
        if "conv2d" in i:
            conv_layer(model)
        elif "dense_layer" in i:
            dense_layer(model)
        elif "drop_layer" in i:
            dropout_layer(model)
        elif "max_layer" in i:
            maxpooling_layer(model)
        elif "active_layer" in i:
            activation_layer(model)
        elif "average_layer" in i:
            averagepooling_layer(model)
        elif "flat_layer" in i:
            flatten_layer(model)
        elif "output_layer"in i:
            output_layer(model)
        elif "build_layer"in i:
            build_model()
        elif "summary_layer"in i:
            run_model()
        elif "train_layer"in i:
            history = fit_model(model, X_train, y_train, X_val, y_val)
            print(history)
        elif "preprocess_layer"in i:
            X_train, X_val, y_train, y_val = preprocess_and_build_model()"""
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
