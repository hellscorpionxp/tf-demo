'''
Created on 2020年3月24日

@author: tony
'''
import tensorflow.compat.v1 as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

tf.disable_v2_behavior()

digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)


def add_layer(inputs, in_size, out_size, activation_function=None):
  weights = tf.Variable(tf.random_normal([in_size, out_size]))
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
  wx_plus_b = tf.matmul(inputs, weights) + biases
  wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)
  if activation_function is None:
    outputs = wx_plus_b
  else:
    outputs = activation_function(wx_plus_b)
  return outputs


keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

layer1 = add_layer(xs, 64, 50, activation_function=tf.nn.tanh)
prediction = add_layer(layer1, 50, 10, activation_function=tf.nn.softmax)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

with tf.Session() as sess:
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('logs/train', sess.graph)
  test_writer = tf.summary.FileWriter('logs/test', sess.graph)
  sess.run(tf.initialize_all_variables())
  for i in range(500):
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
      train_result = sess.run(merged, feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
      test_result = sess.run(merged, feed_dict={xs: x_test, ys: y_test, keep_prob: 1})
      train_writer.add_summary(train_result, i)
      test_writer.add_summary(test_result, i)
