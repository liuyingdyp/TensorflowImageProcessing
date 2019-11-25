import os
import tensorflow as tf
from PIL import Image # 处理图像的包
import numpy as np

folder = 'image/'
# 设定图像类别标签，标签名称和对应目录名相同
label = {'cat', 'dog'}
# 要生成的文件
writer = tf.python_io.TFRecordWriter(folder + 'cat_dog.tfrecords')
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
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))#example对象对label和image数据进行封装；Int64List存放图像的标签对应的整数；BytesList存放图像数据
        writer.write(example.SerializeToString())  # 序列化为字符串
        count = count+1
writer.close()

# 定义数据流队列
filename_queue = tf.train.string_input_producer([folder+'cat_dog.tfrecords'])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue) #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   })# 取出包含image和label的features对象，拆解为图像数据和标签数据
# bytes:  字符串类型的张量。所有元素的长度必须相同
# out_type:   来自tf.half，tf.float32，tf.float64，tf.int32，tf.uint16，tf.uint8，tf.int16，tf.int8，tf.int64的tf.DType
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [128, 128, 3])
label = tf.cast(features['label'], tf.int32)
# 开始一个会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 协同启动的线程；调用 tf.train.Coordinator() 来创建一个线程协调器，用来管理之后在Session中启动的所有线程;
    coord = tf.train.Coordinator()
    # 以多线程的方式启动对队列的处理
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(count):
        example, l = sess.run([image, label])# 在会话中取出image和label
        img = Image.fromarray(example, 'RGB')# from PIL import Image 的引用
        img.save(folder + str(i) + '_Label_' + str(l) + '.jpg')# 存下图片
        print(example, l)
    coord.request_stop()# 请求停止线程
    coord.join(threads)# 等待所有进程结束