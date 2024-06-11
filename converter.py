import tensorflow as tf

model = tf.keras.models.load_model('1D-CNN_model_HDF5.h5')  # Load your trained model
tf.saved_model.save(model, '1D-CNN_model_HDF5_2_Savemodel')

converter = tf.lite.TFLiteConverter.from_saved_model('1D-CNN_model_HDF5_2_Savemodel')
tflite_model2 = converter.convert()
with open('model2.tflite', 'wb') as f:
    f.write(tflite_model2)

converter = tf.lite.TFLiteConverter.from_saved_model('1D-CNN_model_SavedModel')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


