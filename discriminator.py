import tensorflow as tf
from Unet import UNet

class Discriminator():
    def __init__(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(image, filters=64, kernel_size=[3,3], strides=(2,2), padding='SAME',name='d_h0_conv')))
            h1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(h0, filters=128, kernel_size=[3,3], strides=(2,2), padding='SAME', name='d_h1_conv')))
            h2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(h1, filters=256, kernel_size=[3,3], strides=(2,2),padding='SAME', name='d_h2_conv')))
            h3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(h2, filters=512, kernel_size=[3,3], strides=(2,2),padding='SAME', name='d_h3_conv')))
            print(h3.get_shape().as_list())
            self.last_h = tf.layers.dense(tf.reshape(h3, [-1,32*32*512]), 1, name='d_dense_layer')
            self.out = tf.nn.sigmoid(self.last_h,name='out')
