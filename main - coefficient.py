import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定使用0块卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1;0

import tensorflow as tf
# from tensorflow.keras import layers, models
import time
import datetime
from keras import layers, models
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)
print("CUDA version used by TensorFlow:", tf.test.is_built_with_cuda())
print("GPU support:", tf.test.is_built_with_gpu_support())  #########################

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if not gpus:
    print("No GPU devices found.")
else:
    for gpu in gpus:
        # tf.config.experimental.set_memory_growth(gpu, True)
        # print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        # print("GPU device name:", gpu.name)
        # print("GPU device description:", gpu.device_type)
        # print("GPU device memory:", tf.test.gpu_device_name())
        print("GPU device details:", tf.config.experimental.get_device_details(gpu))

    # 选择第一块 GPU
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # gpus[0] ; []

# 从MATLAB加载已训练好的参数
# 获取参数值,当前获取的为字典类型
weight_values_dict = scipy.io.loadmat('new_weights.mat')  # 假设参数在MATLAB文件中被命名为 'weights',是字典类型
bias_values_dict = scipy.io.loadmat('new_biases.mat')  # 假设参数在MATLAB文件中被命名为 'biases'，是字典类型

# 单独将权重和偏差取出来，当前获取的为一个包含多个NumPy数组的嵌套列表（Nested List）
weights_list = weight_values_dict['weights']
biases_list = bias_values_dict['biases']

# 逐个取出数组
biases_array = []
for n_array in biases_list:
    for array in n_array:
        biases_array.append(array)

weights_array = []
for n_array in weights_list:
    for array in n_array:
        weights_array.append(array)

# 分配参数值给对应的变量
'''
##查看此时神经网络需要的参数形状，以匹配参数格式
var = network.layers[13].get_weights()[0].shape  
print("network.layers[13].get_weights()[0].shape= ",var)
'''


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

    network = models.Sequential([
        # 添加一维卷积层
        layers.Conv1D(filters=4, kernel_size=6, activation='LeakyReLU', input_shape=input_shape),
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
        # layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(num_class, activation='softmax')
    ])
    # network.summary()

    ##将加载的参数写进神经网络
    network.layers[0].set_weights([weights_array[1], biases_array[1][0]])
    network.layers[2].set_weights([weights_array[4], biases_array[4][0]])
    network.layers[4].set_weights([weights_array[7], biases_array[7][0]])
    network.layers[6].set_weights([weights_array[10], biases_array[10][0]])
    network.layers[8].set_weights([weights_array[13], biases_array[13][0]])
    network.layers[10].set_weights([weights_array[16], biases_array[16][0]])
    network.layers[13].set_weights([tf.transpose(weights_array[20]), tf.transpose(
        biases_array[20].reshape(-1))])  ## weights.shape=(8, 50) Bias.shape=(50,)
    network.layers[14].set_weights([tf.transpose(weights_array[21]), tf.transpose(biases_array[21].reshape(-1))])
    network.layers[15].set_weights([tf.transpose(weights_array[22]), tf.transpose(biases_array[22].reshape(-1))])
    '''
    ##验证是否正确写入weights和biases
    print("Fully connected weights: "，network.layers[15].weights[0])
    print("Fully connected biases: "，network.layers[15].weights[1])
    '''
    ##编译模型
    network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], )

    ##加载数据
    # 1.导入数据

    ## 使用
    folder_path_B_O = r'D:\pyCharm_work\createCNN_coefficient\database\environmentalSimulationDatabase\noise\newData-0dB\B_O-0dB'
    folder_path_D_F = r'D:\pyCharm_work\createCNN_coefficient\database\environmentalSimulationDatabase\noise\newData-0dB\D_F-0dB'
    folder_path_E_S = r'D:\pyCharm_work\createCNN_coefficient\database\environmentalSimulationDatabase\noise\newData-0dB\E_S-0dB'
    data_array_B_O = read_txt(folder_path_B_O)
    data_array_D_F = read_txt(folder_path_D_F)
    data_array_E_S = read_txt(folder_path_E_S)

    combined_data = np.concatenate((data_array_B_O[:, :, :], data_array_D_F[:, :, :], data_array_E_S[:, :, :]), axis=0)

    label_array_B_O = np.zeros(100)
    label_array_D_F = np.ones(100)
    label_array_E_S = np.full((100,), 2)

    combined_label = np.concatenate((label_array_B_O[:], label_array_D_F[:], label_array_E_S[:]), axis=0)
    # print(combined_data.shape)  # 输出 (文件数量, 行数, 列数)
    # print(combined_label.shape)

    x_train, x_test, y_train, y_test = train_test_split(combined_data, combined_label, test_size=0.3, random_state=42)
    ##(210, 4097, 1) (90, 4097, 1) (210,) (90,)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # 使用数据验证模型
    # X_val 是验证集特征数据，形状为 (num_samples, 10, 1)
    # y_val 是验证集标签数据，形状为 (num_samples,)

    # 获取当前时间
    start_time_ = time.time()
    start_time_v = datetime.datetime.now()
    formatted_time_start = start_time_v.strftime('%H:%M:%S.%f')[:-3]
    print("Start time: ", formatted_time_start)

    validation_loss, validation_accuracy = network.evaluate(x=x_test, y=y_test,batch_size=32,verbose=0)# use default batch_size=32 verbose=1

    end_time_ = time.time()
    end_time_v = datetime.datetime.now()
    formatted_time_end = end_time_v.strftime('%H:%M:%S.%f')[:-3]
    print("End time: ", formatted_time_end)

    test_time_v = end_time_ - start_time_
    print("Test time: {:.3f} seconds".format(test_time_v))
    print(f"Test Loss: {validation_loss}")
    print(f"Test Accuracy: {validation_accuracy}")


if __name__ == "__main__":
    main()

# 假设你已经有用于验证的数据集
