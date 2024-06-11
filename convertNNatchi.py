import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#from tensorflow.keras import layers, models

from keras import layers, models

import numpy as np

input_shape = (4097, 1)
network = models.Sequential([
 # 添加一维卷积层
layers.Conv1D(filters=4, kernel_size=6, activation='LeakyReLU', input_shape= input_shape),
layers.MaxPooling1D(pool_size=2),
layers.Conv1D(filters=4, kernel_size=5, activation='LeakyReLU'),
layers.MaxPooling1D(pool_size=2),
layers.Conv1D(filters=10, kernel_size=4, activation='LeakyReLU'),
layers.MaxPooling1D(pool_size=2),
layers.Conv1D(filters=10, kernel_size=4, activation='LeakyReLU'),
layers.MaxPooling1D(pool_size=2),
layers.Conv1D(filters=15, kernel_size=4, activation='LeakyReLU'),
layers.MaxPooling1D(pool_size=2),
layers.Conv1D(filters=8, kernel_size=6, activation='relu'),
layers.LayerNormalization(),
layers.GlobalAvgPool1D(),
        # 添加全连接层
        #layers.Flatten(),
layers.Dense(50, activation='relu'),
layers.Dense(20, activation='relu'),
layers.Dense(3, activation= 'softmax')
])
# network.summary()

##编译模型
network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], )

network.save('1D-CNN_archi_HDF5.h5') ##保存为HDF5格式
tf.saved_model.save(network,'1D-CNN_archi_SavedModel')

model = tf.keras.models.load_model('1D-CNN_archi_HDF5.h5')  # Load your trained model


converter = tf.lite.TFLiteConverter.from_saved_model('1D-CNN_archi_SavedModel')
tflite_model_CNN_archi = converter.convert()
with open('CNN_archi.tflite', 'wb') as f:
    f.write(tflite_model_CNN_archi)

