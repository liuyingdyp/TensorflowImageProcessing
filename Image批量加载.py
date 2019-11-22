import tensorflow as tf
path = 'cat.jpeg'
# 创建输入队列
file_queue = tf.train.string_input_producer([path])
image_reader = tf.WholeFileReader()
_, image = image_reader.read(file_queue)
image = tf.image.decode_jpeg(image)

with tf.Session() as sess:
    # 协同启动的线程
    coord = tf.train.Coordinator()
    # 启动线程运行队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(image))
    # 停止所有线程
    coord.request_stop()
    coord.join(threads)
