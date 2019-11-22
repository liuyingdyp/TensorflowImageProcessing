import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data_ipg = tf.gfile.FastGFile('cat.jpeg','rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data_ipg) # 图像解码
    plt.figure(1) # 图像显示
    print(sess.run(img_data)) # 显示图像矩阵
    plt.imshow(img_data.eval()) #显示图像



