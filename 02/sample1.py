'''
Created on 2020年3月19日

@author: tony
'''
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

with tf.Session() as sess:
  result = sess.run(product)
  print(result)