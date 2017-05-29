import tensorflow as tf  
import numpy as np

class UNet():
    def __init__(self, input_l):
        #if you haven't known about UNet yet,check out following.
        #https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

        #input_layer
        #batch_size, x, y, RGB

        enc_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_l, filters=64, kernel_size=[3,3], activation=tf.nn.relu))
        enc_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv1, filters=64, kernel_size=[3,3], activation=tf.nn.relu))

        enc_pool1 = tf.layers.max_pooling2d(inputs=enc_conv1, pool_size=[2,2], strides=2)

        enc_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool1, filters=128, kernel_size=[3,3], activation=tf.nn.relu))
        enc_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv2, filters=128, kernel_size=[3,3], activation=tf.nn.relu))

        enc_pool2 = tf.layers.max_pooling2d(inputs=enc_conv2, pool_size=[2,2], strides=2)

        enc_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool2, filters=256, kernel_size=[3,3], activation=tf.nn.relu))
        enc_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv3, filters=256, kernel_size=[3,3], activation=tf.nn.relu))

        enc_pool3 = tf.layers.max_pooling2d(inputs=enc_conv3, pool_size=[2,2], strides=2)

        enc_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool3, filters=512, kernel_size=[3,3], activation=tf.nn.relu))
        enc_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv4, filters=512, kernel_size=[3,3], activation=tf.nn.relu))
        
        enc_pool4 = tf.layers.max_pooling2d(inputs=enc_conv4, pool_size=[2,2], strides=2)
        
        enc_conv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool4, filters=1024, kernel_size=[3,3], activation=tf.nn.relu))
        enc_conv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv5, filters=1024, kernel_size=[3,3], activation=tf.nn.relu))

        def copy_and_crop(inputs, sx, ex, sy, ey):
            print(inputs.get_shape().as_list())  
            #inputs_copy = tf.Variable(inputs)
            return inputs[:, sx:ex, sy:ey, :]
       
        concat_conv1 = copy_and_crop(enc_conv4, 4, -4, 4, -4)
        upconv1 = tf.layers.conv2d_transpose(inputs=enc_conv5, filters=512, kernel_size=[2,2],strides=(2,2))
        upconv1 = tf.concat([concat_conv1 ,upconv1], 3)

        dec_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv1, filters=512, kernel_size=[3,3], activation=tf.nn.relu))
        dec_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv1, filters=512, kernel_size=[3,3], activation=tf.nn.relu))

        concat_conv2 = copy_and_crop(enc_conv3, 16, -16, 16, -16)
        upconv2 = tf.layers.conv2d_transpose(inputs=dec_conv1, filters=256, kernel_size=[2,2],strides=(2,2))
        upconv2 = tf.concat([concat_conv2 ,upconv2], 3)

        dec_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv2, filters=256, kernel_size=[3,3], activation=tf.nn.relu))
        dec_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv2, filters=256, kernel_size=[3,3], activation=tf.nn.relu))

        concat_conv3 = copy_and_crop(enc_conv2, 40, -40, 40, -40)
        upconv3 = tf.layers.conv2d_transpose(inputs=dec_conv2, filters=128, kernel_size=[2,2],strides=(2,2))
        upconv3 = tf.concat([concat_conv3, upconv3], 3)

        dec_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv3, filters=128, kernel_size=[3,3], activation=tf.nn.relu))
        dec_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv3, filters=128, kernel_size=[3,3], activation=tf.nn.relu))

        concat_conv3 = copy_and_crop(enc_conv1, 88, -88, 88, -88)
        upconv4 = tf.layers.conv2d_transpose(inputs=dec_conv3, filters=64, kernel_size=[2,2], strides=(2,2))
        upconv4 = tf.concat([concat_conv3, upconv4], 3)

        dec_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv4, filters=64, kernel_size=[3,3], activation=tf.nn.relu))
        dec_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv4, filters=64, kernel_size=[3,3], activation=tf.nn.relu))

        self.dec_conv_last = tf.layers.batch_normalization(tf.layers.conv2d(inputs= dec_conv4, filters=3, kernel_size=[1,1]))
