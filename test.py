
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# IMPORTING THE TRAINED MODEL
model = tf.keras.models.load_model('F:/brain-tumor-classifier-app-master (2)/brain-tumor-classifier-app-master/f_model.h5')

# GETTING ALL THE IMAGES FROM SAMPLES FOLDER
data_dir = 'F:/brain-tumor-classifier-app-master (2)/brain-tumor-classifier-app-master/samples'
images = []
for img in os.listdir(data_dir):
    img = os.path.join(data_dir, img)
    img = tf.keras.preprocessing.image.load_img(img, target_size=(200, 200), color_mode="grayscale")
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
files = os.listdir(data_dir)
images = np.vstack(images)

# PREDICTING THE CLASSIFICATION FOR EVERY FILE IN THE SAMPLES DIRECTORY
classes = model.predict(images, batch_size=1)

# USER UNDERSTANDABLE OUTPUT FORMATTING
print(classes)