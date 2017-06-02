import tensorflow as tf
from train import Train
import os
from PIL import Image
import numpy as np
from time import time

output_dir = './test_o'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def image_read(file_name_path, size=512, channel=3):
    imgs = np.empty((0,size,size,channel), dtype=np.float32)
    img = np.array(Image.open(file_name_path)).astype(np.float32)
    imgs = np.append(imgs, np.array([img]))
    return imgs.reshape((-1, size, size, channel))

def image_save(img, dir_path=output_dir):
    print(len(img[0]))
    img = Image.fromarray(np.uint8(img[0]))
    img.save(dir_path + {}+".jpg".format(time().split('.')[0]))

def main():
    train = Train()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "saved/model.ckpt")
        
        img = image_read('./data/linedraw512/24f10f48f37f376972.jpg')
        fakes = sess.run([train.fakeA],{train.realB:img})
        image_save(fakes)
        
if __name__ == '__main__':
    main()
