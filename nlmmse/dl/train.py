import numpy as np
import scipy.io
from tensorflow import keras
from tensorflow.keras import layers


filepath = 'data/1/data_1.mat'

f = scipy.io.loadmat(filepath)

x = np.array(f['x'])
t = np.array(f['t'])

x = np.reshape(x, x.shape + (1,))
t = np.reshape(t, t.shape + (1,))

input_shape = 10

model = keras.Sequential(
    [
        keras.Input(shape=(input_shape, 1)),
        layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'),
        layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.GlobalMaxPooling1D(),
        layers.Flatten(),
        layers.Dense(input_shape, activation="softmax"),
    ]
)

print(model.summary())

batch_size = 16
epochs = 50

model.compile(loss="mse", optimizer="adam", metrics=[keras.metrics.MeanAbsoluteError()])

model.fit(x, t, batch_size=batch_size, epochs=epochs, validation_split=0.3)
