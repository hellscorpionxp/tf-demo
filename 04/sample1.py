'''
Created on 2020年3月19日

@author: tony
'''
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
  print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))