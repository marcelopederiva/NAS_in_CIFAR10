import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def gets_model(batch_size, epochs, channel1= 32, Kernel1 = 3, channel2 = 64, Kernel2 = 3, dropout=0.5, activation="relu"):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(channel1, kernel_size=(Kernel1, Kernel1), activation=activation,padding='same'),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(channel2, kernel_size=(Kernel2, Kernel2), activation=activation,padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def evaluate_in_test(batch_size=64, epochs=15, channel1= 32, Kernel1 = 3, channel2 = 64, Kernel2 = 3, dropout=0.5, activation="relu"):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='val_accuracy.hdf5',
                                                                    save_weights_only=True,
                                                                    monitor='val_accuracy',
                                                                    mode='max',
                                                                    save_best_only=True)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    model = gets_model(batch_size, epochs, channel1= channel1, Kernel1 = Kernel1, channel2 = channel2,
                        Kernel2 = Kernel2, dropout=dropout, activation="relu")

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    hist = model.fit(x_train, y_train, batch_size=batch_size,
                     callbacks=[model_checkpoint_callback,callback], epochs=epochs, validation_split=0.1)
    model.load_weights('val_accuracy.hdf5')
    score = model.evaluate(x_test, y_test, verbose=0)
    model.summary()
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score

def training_n_out(batch_size, epochs, channel1= 32, Kernel1 = 3, channel2 = 64, Kernel2 = 3, dropout=0.5, activation="relu"):
    model = gets_model(batch_size, epochs, channel1= channel1, Kernel1 = Kernel1, channel2 = channel2,
                        Kernel2 = Kernel2, dropout=dropout, activation="relu")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
    score = np.max(hist.history['val_accuracy'])
    return score
    
if __name__=='__main__':

    s = evaluate_in_test(epochs=15,
                        batch_size=64,
                        channel1=round(32),
                        channel2=round(64),
                        Kernel1=round(3),
                        Kernel2=round(3),
                        dropout=np.round(0.5,decimals=2))

   

