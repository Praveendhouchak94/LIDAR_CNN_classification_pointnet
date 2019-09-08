from keras.layers.core import Reshape
from keras.layers.pooling import MaxPooling2D
import tensorflow as tf
from keras.models import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
import numpy as np


class PointNet:

    @staticmethod
    def build():
        """ Classification PointNet, input is BxNx3, output Bx40 """
        num_point = 2048
        point_cloud = Input((2048, 3))
        net = Reshape(target_shape=(2048, 3, 1))(point_cloud)

        net = Conv2D(64, (1, 3), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = MaxPooling2D((num_point, 1), padding='valid')(net)
        net = Reshape(target_shape=(1024,))(net)
        net = Dense(512, activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dense(256, activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(net)
        transform_input = Reshape(target_shape=(3, 3))(net)

        point_cloud_transformed = Lambda(lambda x: tf.matmul(x[0], x[1]))([point_cloud, transform_input])

        input_image = Reshape(target_shape=(2048, 3, 1))(point_cloud_transformed)

        net = Conv2D(64, (1, 3), strides=(1, 1), padding="valid", activation='relu')(input_image)
        net = BatchNormalization()(net)
        net = Conv2D(64, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net_forward = BatchNormalization()(net)
        net = Reshape(target_shape=(2048, 1, 64))(net_forward)

        net = Conv2D(64, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)

        net = MaxPooling2D((num_point, 1), padding='valid')(net)
        print(net)
        net = Reshape(target_shape=(1024,))(net)
        net = Dense(512, activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dense(256, activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(net)
        transform_input = Reshape((64, 64))(net)

        net_transformed = Lambda(lambda x: tf.matmul(x[0], x[1]))([Reshape((2048, 64))(net_forward), transform_input])

        net = Reshape(target_shape=(2048, 1, 64))(net_transformed)
        net = Conv2D(64, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid", activation='relu')(net)
        net = BatchNormalization()(net)
        net = MaxPooling2D((num_point, 1), padding='valid')(net)

        net = Reshape(target_shape=(1024,))(net)
        net = Dense(512, activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dropout(rate=0.7)(net)
        net = Dense(256, activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dropout(rate=0.7)(net)
        net = Dense(40, activation='softmax')(net)

        model = Model(inputs=point_cloud, outputs=net)
        return model
