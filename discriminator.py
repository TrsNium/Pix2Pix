import tensorflow as tf
from Unet import UNet

def Discriminator():
    def __init__(self, image, batch_size):
        h0 = tf.nn.relu(tf.layers.conv2d(image, filters=32, kernel_size=[3,3]))
        h1 = tf.nn.relu(tf.layers.conv2d(h0, filters=64, kernel_size=[3,3]))
        h2 = tf.nn.relu(tf.layers.conv2d(h1, filters=64, kernel_size=[3,3]))
        h3 = tf.nn.relu(tf.layers.conv2d(h2, filters=64, kernel_size=[3,3]))
        h4 = tf.layers.dense(tf.reshape(h3, [batch_size, -1]), 1)
        self.out = tf.nn.sigmoid(h4)
