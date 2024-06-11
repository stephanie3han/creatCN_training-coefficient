import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 指定使用0块卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1;启用GPU 0

import tensorflow as tf
#from tensorflow.keras import layers, models
import time
from tensorflow import keras
from keras import layers, models
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

#profiler = tf.profiler.experimental.Profile(tf.profiler.experimental.ProfilerOptions(host_tracer_level=2, device_tracer_level=2))

print("TensorFlow version:", tf.__version__)
print("CUDA version used by TensorFlow:", tf.test.is_built_with_cuda())
print("GPU support:", tf.test.is_built_with_gpu_support()) #########################


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if not gpus:
    print("No GPU devices found.")
else:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        print("GPU device name:", gpu.name)
        print("GPU device description:", gpu.device_type)
        print("GPU device memory:", tf.test.gpu_device_name())
        print("GPU device details:", tf.config.experimental.get_device_details(gpu))

    # 选择第一块 GPU
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU') #gpus[0] ; []


# 获取文件夹中所有 TXT 文件的文件名列表
def read_txt(path: object) -> object:
    file_names = [file for file in os.listdir(path) if file.endswith('.txt')]
    # 初始化一个空数组，用于存储所有数据
    data_array = np.empty((0, 4097, 1))
    # 逐个读取并处理每个 .txt 文件
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            file_data = np.array([float(line.strip()) for line in lines]).reshape((-1, 1))
            file_data = np.expand_dims(file_data, axis=0)
            data_array = np.vstack((data_array, file_data))
    return data_array


def main():
    ##建立模型
    input_shape = (4097, 1)
    num_class = 3

    network = keras.Sequential([
        # 添加一维卷积层
        keras.layers.Conv1D(filters=4, kernel_size=6, activation='LeakyReLU', input_shape=(4097,1)),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=4, kernel_size=5, activation='LeakyReLU'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=10, kernel_size=4, activation='LeakyReLU'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=10, kernel_size=4, activation='LeakyReLU'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=15, kernel_size=4, activation='LeakyReLU'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=8, kernel_size=6, activation='relu'),
        keras.layers.LayerNormalization(),
        keras.layers.GlobalAvgPool1D(),
        # 添加全连接层
        #layers.Flatten(),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(num_class, activation='softmax')
    ])
    # network.summary()



    ##导出模型
    converter = tf.lite.TFLiteConverter.from_keras_model(network)
    tflite_model = converter.convert()

    # 将 TFLite 模型保存到文件
    with open('network_model.tflite', 'wb') as f:
        f.write(tflite_model)


    #compile
    network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], )


    ##加载数据
    # 1.导入数据

    ## 使用
    #folder_path_B_O = r'D:\pyCharm_work\CNN_training_data_interference\database\environmentalSimulationDatabase\noise\newData-0dB\B_O-0dB'
   # folder_path_D_F = r'D:\pyCharm_work\CNN_training_data_interference\database\environmentalSimulationDatabase\noise\newData-0dB\D_F-0dB'
    #folder_path_E_S = r'D:\pyCharm_work\CNN_training_data_interference\database\environmentalSimulationDatabase\noise\newData-0dB\E_S-0dB'
    folder_path_B_O = r'D:\pyCharm_work\CNN_training_data_interference\database\newData\B_O'
    folder_path_D_F = r'D:\pyCharm_work\CNN_training_data_interference\database\newData\D_F'
    folder_path_E_S = r'D:\pyCharm_work\CNN_training_data_interference\database\newData\E_S'
    data_array_B_O = read_txt(folder_path_B_O)
    data_array_D_F = read_txt(folder_path_D_F)
    data_array_E_S = read_txt(folder_path_E_S)
    
    combined_data = np.concatenate((data_array_B_O[:, :, :], data_array_D_F[:, :, :], data_array_E_S[:, :, :]), axis=0)
   
    #combined_data_validation = np.concatenate((data_array_B_O[85:100,:,:], data_array_D_F[85:100,:,:], data_array_E_S[85:100,:,:]), axis=0)
    #combined_data_test = np.concatenate((data_array_B_O[70:85, :, :], data_array_D_F[70:85, :, :], data_array_E_S[70:85, :, :]), axis=0)

    label_array_B_O = np.zeros(100)
    label_array_D_F = np.ones(100)
    label_array_E_S = np.full((100,),2)
    
    combined_label= np.concatenate((label_array_B_O[:], label_array_D_F[:], label_array_E_S[:]),axis=0)
    
    #combined_label_validation = np.concatenate((label_array_B_O[85:], label_array_D_F[85:], label_array_E_S[85:]), axis=0)
    #combined_label_test = np.concatenate((label_array_B_O[70:85], label_array_D_F[70:85], label_array_E_S[70:85]), axis=0)
    #print(combined_data.shape)  # 输出 (文件数量, 行数, 列数)
    #print(combined_label.shape)

    ## 交叉验证
    x_train, x_test, y_train, y_test = train_test_split(combined_data, combined_label, test_size=0.3, random_state=42)
    ##(210, 4097, 1) (90, 4097, 1) (210,) (90,)
    '''
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs', histogram_freq=0, write_graph=True,
        write_images=False, update_freq='epoch', profile_batch=2,
        embeddings_freq=0, embeddings_metadata=None
    )  ## 加入该语句
    '''

    start_time = time.time()
    # 使用strftime将时间戳转换为人类可读的格式
    formatted_time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time: ",formatted_time_start)
    ##训练模型
    network.fit(x_train, y_train, batch_size=3, epochs=100, validation_data=(x_test, y_test))  ####  150  #####, callbacks=[tboard_callback]
    #其中，x_train是训练数据，y_train是训练标签，x_val是验证数据，y_val是验证标签。batch_size表示每次训练的批次大小，epochs表示训练的轮数
    #network.save('1D-CNN_model_HDF5.h5') ##保存为HDF5格式
    #tf.saved_model.save(network,'1D-CNN_model_SavedModel')

    end_time = time.time()
    formatted_time_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End time: ",formatted_time_end)
    training_time = end_time - start_time
    # 输出训练时间
    print("Training time: {:.2f} seconds".format(training_time))

    # 使用数据验证模型
    # X_val 是验证集特征数据，形状为 (num_samples, 10, 1)
    # y_val 是验证集标签数据，形状为 (num_samples,)
    #validation_loss, validation_accuracy = network.evaluate(combined_data_test, combined_label_test)
    start_time_v = time.time()
    validation_loss, validation_accuracy = network.evaluate(x_test, y_test)
    end_time_v = time.time()
    training_time_v = end_time_v - start_time_v
    print("Validate time: {:.2f} seconds".format(training_time_v))
    ##使用模型进行预测
    #predictions = network.predict(x_test)
    #model.evaluate() 用于评估模型在给定测试数据上的性能，返回损失和指标值。
    #model.predict() 用于对给定输入数据进行预测，返回模型的输出结果。

    print(f"Validation Loss: {validation_loss}")
    print(f"Validation Accuracy: {validation_accuracy}")

if __name__=="__main__":
    main()





# 假设你已经有用于验证的数据集


