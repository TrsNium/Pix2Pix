import tensorflow as tf
import numpy as np

class UNet():
    def __init__(self, feature):
        #if you haven't known about UNet yet,check out following.
        #https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

        #input_layer
        #batch_size, x, y, RGB
        input_l = tf.reshape(feature, [-1, 572, 572, 3])

        conv1 = tf.layers.conv2d(inputs=input_l, filters=64, kernel_size=[3,3,3], activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3,3], activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3,3], activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3,3], activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

        conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3,3], activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3,3], activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3,3], strides=2)

        conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3,3], activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[3,3], activation=tf.nn.relu)
        
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)

