import tensorflow as tf
from Unet import UNet

class Discriminator():
    def __init__(self, image, batch_size):
        h0 = tf.nn.relu(tf.layers.conv2d(image, filters=32, kernel_size=[3,3]))
        h1 = tf.nn.relu(tf.layers.conv2d(h0, filters=64, kernel_size=[3,3]))
        h2 = tf.nn.relu(tf.layers.conv2d(h1, filters=64, kernel_size=[3,3]))
        h3 = tf.nn.relu(tf.layers.conv2d(h2, filters=64, kernel_size=[3,3]))
#        print(h3.get_shape().as_list())
        self.last_h = tf.layers.dense(tf.reshape(h3, [-1, 380*380*64]), 1)
        self.out = tf.nn.sigmoid(self.last_h)
