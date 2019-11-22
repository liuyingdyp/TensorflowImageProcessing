import os
import tensorflow as tf
from PIL import Image # 处理图像的包
import numpy as np

folder = 'image/'
# 设定图像类别标签，标签名称和对应目录名相同
label = {'cat', 'dog'}
# 要生成的文件
writer = tf.python_io.TFRecordWriter(folder + 'cat_dog.tfrecord')
# 记录图像的个数
count = 0
for index, name in enumerate(label):
    folder_path = folder + name + '/'
    for img_name in os.listdir(folder_path):
        img_path = folder_path + img_name# 在image/cat、dog文件夹里的每一张照片的完整访问路径
        img = Image.open(img_path)# 获取img_path路径下的图片
        img = img.resize((128, 128))# 设置图片大小width=128；height=128
        img_raw = img.tobytes()# 将图片转化为二进制格式
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value={[index]})),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))#example对象对label和image数据进行封装；Int64List存放图像的标签对应的整数；BytesList存放图像数据
    writer.write(example.SerializeToString())  # 序列化为字符串
