import tensorflow as tf  
import numpy as np

class UNet():
    def __init__(self, input_l):
        #if you haven't known about UNet yet,check out following.
        #https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

        #input_layer
        #batch_size, x, y, RGB

        enc_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_l, filters=64, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv1'))
        enc_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv1, filters=64, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv2'))

        enc_pool1 = tf.layers.max_pooling2d(inputs=enc_conv1, pool_size=[2,2], strides=2, name='g_enc_pool1')

        enc_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool1, filters=128, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv3'))
        enc_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv2, filters=128, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv4'))

        enc_pool2 = tf.layers.max_pooling2d(inputs=enc_conv2, pool_size=[2,2], strides=2, name= 'g_enc_pool2')

        enc_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool2, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv5'))
        enc_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv3, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv6'))

        enc_pool3 = tf.layers.max_pooling2d(inputs=enc_conv3, pool_size=[2,2], strides=2, name='g_enc_pool3')

        enc_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool3, filters=512, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv7'))
        enc_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv4, filters=512, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv8'))
        
        enc_pool4 = tf.layers.max_pooling2d(inputs=enc_conv4, pool_size=[2,2], strides=2, name='g_enc_pool4')
        
        enc_conv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_pool4, filters=1024, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv9'))
        enc_conv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_conv5, filters=1024, kernel_size=[3,3], activation=tf.nn.relu, name='g_enc_conv10'))

        def copy_and_crop(inputs, sx, ex, sy, ey):
            print(inputs.get_shape().as_list())  
            #inputs_copy = tf.Variable(inputs)
            return inputs[:, sx:ex, sy:ey, :]
       
        concat_conv1 = copy_and_crop(enc_conv4, 4, -4, 4, -4)
        upconv1 = tf.layers.conv2d_transpose(inputs=enc_conv5, filters=512, kernel_size=[2,2],strides=(2,2), name='g_dec_dc0')
        upconv1 = tf.concat([concat_conv1 ,upconv1], 3)

        dec_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv1, filters=512, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc1'))
        dec_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv1, filters=512, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc3'))

        concat_conv2 = copy_and_crop(enc_conv3, 16, -16, 16, -16)
        upconv2 = tf.layers.conv2d_transpose(inputs=dec_conv1, filters=256, kernel_size=[2,2],strides=(2,2), name='g_dec_dc4')
        upconv2 = tf.concat([concat_conv2 ,upconv2], 3)

        dec_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv2, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc5'))
        dec_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv2, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc6'))

        concat_conv3 = copy_and_crop(enc_conv2, 40, -40, 40, -40)
        upconv3 = tf.layers.conv2d_transpose(inputs=dec_conv2, filters=128, kernel_size=[2,2],strides=(2,2), name='g_dec_dc7')
        upconv3 = tf.concat([concat_conv3, upconv3], 3)

        dec_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv3, filters=128, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc8'))
        dec_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv3, filters=128, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc9'))

        concat_conv4 = copy_and_crop(enc_conv1, 88, -88, 88, -88)
        upconv4 = tf.layers.conv2d_transpose(inputs=dec_conv3, filters=64, kernel_size=[2,2], strides=(2,2), name='g_dec_dc10')
        upconv4 = tf.concat([concat_conv4, upconv4], 3)

        dec_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=upconv4, filters=64, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc11'))
        dec_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=dec_conv4, filters=64, kernel_size=[3,3], activation=tf.nn.relu, name='g_dec_dc11'))

        self.dec_conv_last = tf.layers.conv2d(inputs= dec_conv4, filters=3, kernel_size=[1,1], name='g_dec_dc12')

class UNet1():
    def __init__(self,inputs):
        enc_c0 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_enc_c0'),name='g_bn_c0'))
        enc_c1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c0, filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_enc_c1'),name='g_bn_c1'))
        enc_c2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c1, filters=64, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_enc_c2'),name='g_bn_c2'))
        enc_c3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c2, filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_enc_c3'),name='g_bn_c3'))
        enc_c4 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c3, filters=128, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_enc_c4'),name='g_bn_c4'))
        enc_c5 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c4, filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_enc_c5'),name='g_bn_c5'))
        enc_c6 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c5, filters=256, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_enc_c6'),name='g_bn_c6'))
        enc_c7 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c6, filters=512, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_enc_c7'),name='g_bn_c7'))
        enc_c8 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=enc_c7, filters=512, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_enc_c8'),name='g_bn_c8'))

        dec_dc8 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c7,enc_c8],3), filters=512, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_dec_dc8'),name='g_bn_d8'))
        dec_dc7 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc8, filters=256, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_dec_dc7'),name='g_bn_d7'))
        dec_dc6 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c6,dec_dc7],3), filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_dec_dc6'),name='g_bn_d6'))
        dec_dc5 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc6, filters=128, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_dec_dc5'),name='g_bn_d5'))
        dec_dc4 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c4,dec_dc5],3), filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_dec_dc4'),name='g_bn_d4'))
        dec_dc3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc4, filters=64, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_dec_dc3'),name='g_bn_d3'))
        dec_dc2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.concat([enc_c2,dec_dc3],3), filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME', name='g_dec_dc2'),name='b_bn_d2'))
        dec_dc1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dec_dc2, filters=32, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_dec_dc1'),name='g_bn_d1'))
        self.dec_dc0 = tf.layers.conv2d(tf.concat([enc_c0,dec_dc1],3), filters=3, kernel_size=[3,3], strides=(1,1), padding='SAME', name='g_dec_dc0')
