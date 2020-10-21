import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Activation, Flatten
import pandas as pd
import os
import math
import timeit
import sklearn.model_selection as sk

# If the train/test files are generated then no need of train/test split
is_train_test_generated = True
# Choose which network to run and the relevant settings
# Experiment name = mmse-2
#model_name = 'FFNN'
model_name = 'ResNet'

epochs = 100
root = 'data/'
summary =  {}
statsfile = 'data/stats.mat'
stats = scipy.io.loadmat(statsfile)
signal_power = np.array(stats['sig_pow'])

# If GPU is avialble then use distributed strategy
strategy = tf.distribute.MirroredStrategy()
num_devices = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(num_devices))

# Fix the error calculation function
def calculate_error(T, T_hat):
    #   Calculate error
    diff = T-T_hat
    n_samples = diff.shape[0]
    print("No. of testing samples", n_samples)
    diff_norm =  np.linalg.norm(diff)
    error = diff_norm*diff_norm/n_samples
    return error

# https://github.com/hfawaz/dl-4-tsc
def build_resnet_model(input_shape, nb_classes):
    n_feature_maps = 64

    #print(input_shape)

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(nb_classes, activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),  metrics=['mean_absolute_error'])

    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    # file_path = self.output_directory + 'best_model.hdf5'

    # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                        #save_best_only=True)

    # self.callbacks = [reduce_lr] #, model_checkpoint]

    return model

def build_model(input_shape, nb_classes):
    NN_model = Sequential()
    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = input_shape, activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(nb_classes, kernel_initializer='normal',activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()

    return NN_model




# The main method starts
start = timeit.default_timer()

for path, subdirs, files in os.walk(root):
    for name in subdirs:
        total_mse = 0
        for subdirpath, subdirdirs, subdirfiles in os.walk(os.path.join(path, name)):
            data_len_monte_carlo = (len(subdirfiles))
            # print(data_len_mpnte_carlo)
            for datafiles in subdirfiles:
                datafiles = (os.path.join(subdirpath, datafiles))
                #print (datafiles)
                
                f = scipy.io.loadmat(datafiles)
                
                if is_train_test_generated:
                    X_train = np.array(f['x_train'])
                    T_train = np.array(f['t_train'])
                    X_test = np.array(f['x_test'])
                    T_test = np.array(f['t_test'])
                else:
                    X = np.array(f['x'])
                    t = np.array(f['t'])
                    X_train, X_test, T_train, T_test = sk.train_test_split(X,
                                                                        t,
                                                                        test_size=0.3, 
                                                                        random_state = 100)

                print("The train dataset size is:", X_train.shape)
                print("The train dataset label size is:", T_train.shape)
                print("The test dataset size is:", X_test.shape)
                print("The test dataset label size is:", T_test.shape)

                with strategy.scope():
                    print ('******* Using',model_name,'*******')
                    if model_name == 'FFNN':    
                        model = build_model(X_train.shape[1], T_train.shape[1])

                    if model_name == 'ResNet':
                        if len(X_train.shape) == 2:  # if univariate
                            # add a dimension to make it multivariate with one dimension 
                            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                        model = build_resnet_model(X_train.shape[1:], T_train.shape[1])

                    print(model.summary())

                    optimizer = keras.optimizers.Adam()
                    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

                hist = model.fit(X_train, T_train, epochs=epochs, batch_size=256, validation_split = 0.2)

                log = pd.DataFrame(hist.history)

                train_loss = log.loc[log['loss'].idxmin]['loss']
                train_accuracy = log.loc[log['loss'].idxmin]['accuracy']
                
                t_hat = model.predict(X_test)
                
                mse_loss = calculate_error(T_test, t_hat)

                print("The resulting loss for", name, "on dataset", datafiles,"is:", mse_loss)

                total_mse = total_mse + mse_loss
                # print("***************************", total_mse)
            total_mse = total_mse/data_len_monte_carlo
            print ("************Monte Carlo MSE for", name, ":", total_mse)
            FFNN_mse = 10*math.log10(total_mse/signal_power[0, int(name)-1])
            summary[name] = FFNN_mse

data = []
print ("The unsorted list is:", summary)
for key, value in sorted(summary.items(), key=lambda item: int(item[0])):
   data.append(value)
   #print(key, value)

print("Sorted list data is:", data)

stop = timeit.default_timer()

print('Time:', stop - start)  

