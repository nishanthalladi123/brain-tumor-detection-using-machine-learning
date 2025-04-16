import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# IMPORTING IMAGES OF TRAINING SET
dir_train = "D:/brain-tumor-classifier-app-master (2)/archive/Training"

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
train_data = datagen_train.flow_from_directory(dir_train,
                                               target_size=(200, 200),
                                               color_mode='grayscale',
                                               class_mode='categorical',
                                               batch_size=10)

# IMPORTING IMAGES OF TEST SET
dir_test = "D:/brain-tumor-classifier-app-master (2)/archive/Testing"
datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
test_data = datagen_test.flow_from_directory(dir_test,
                                             target_size=(200, 200),
                                             color_mode='grayscale',
                                             class_mode='categorical',
                                             batch_size=10)

def tumorModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(200, 200, 1), activation='relu', filters=32, kernel_size=(5,5)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    return model

ACC_THRESHOLD = 0.97
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > ACC_THRESHOLD:
            print(f"\nReached {ACC_THRESHOLD * 100:.2f}% accuracy, so stopping training!!")
            self.model.stop_training = True

model = tumorModel()
callback = Callback()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    validation_data=test_data,
                    epochs=30,
                    verbose=1,
                    batch_size=1,
                    callbacks=[callback])

# Plot loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Save model
model.save("model_final.h5")
