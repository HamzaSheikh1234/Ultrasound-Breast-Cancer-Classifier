import pandas as pd
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory



# Reproducability
def set_seed(seed=10289):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    '/content/drive/MyDrive/ultrasound breast classification/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    '/content/drive/MyDrive/ultrasound breast classification/val',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

import tensorflow_hub as hub

pretrained_base = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
pretrained_base.trainable = False

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Adding dropout for regularization
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Adding another dropout layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Adding another dropout layer
    layers.Dense(1, activation='sigmoid'),
])

optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy'],
)

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_cb = ModelCheckpoint(
    "/content/drive/MyDrive/ultrasound breast classification/best_model.h5",
    save_best_only=True,
    monitor='val_binary_accuracy',
    mode='max'
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=60,
    callbacks = [checkpoint_cb]
)



history_frame = pd.DataFrame(history.history)
history_frame.loc[15:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();

max_val_binary_accuracy = history_frame['val_binary_accuracy'].max()
epoch_of_max_val_binary_accuracy = history_frame['val_binary_accuracy'].idxmax() + 1

print(f"Highest val_binary_accuracy: {max_val_binary_accuracy} at epoch: {epoch_of_max_val_binary_accuracy}")