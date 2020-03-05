'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
from modelSelector import ModelSelector
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10
epochs = 12

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mnist_trained_model.h5'


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Save model and weights
model_path = os.path.join(save_dir, model_name)
print (model_path)
if os.path.isfile(model_path):
    model = load_model(model_path)
else:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)  
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

score = model.evaluate(x_test, y_test, verbose=0)

print('Overall Test loss:', score[0])
print('Overall Test accuracy:', score[1])

img_rows, img_cols, channels = 28, 28, 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# change n_components depending on percentile or no of components 
model_selector = ModelSelector(img_rows, img_cols, channels, n_components=98, dataset_name='mnist', plot_type='2d', n_plot_sample=5)
x_train_ranked = model_selector.arrange_data(x_train, y_train.flatten())
model_selector.visualize_data(x_train_ranked)
x_test_score, x_test_ranked = model_selector.score_test_data(x_train_ranked, x_test, y_test.flatten())
model_selector.visualize_data(x_test_ranked, data_type='test')
x_test_score_nn = model_selector.nn_score_test_data(model, x_test_ranked)
model_selector.visualize_correlated_inference(x_test_score, x_test_score_nn, "Ranking", "NN")

plt.show()
