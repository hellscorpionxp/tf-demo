'''
Created on 2020年3月20日

@author: tony
'''
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()


def add_layer(n_layer, inputs, in_size, out_size, activation_function=None):
  layer_name = 'layer-%s' % n_layer
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      weights = tf.Variable(tf.random_normal([in_size, out_size]))
      tf.summary.histogram(layer_name + '/weights', weights)
    with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
      tf.summary.histogram(layer_name + '/biases', biases)
    with tf.name_scope('wx_plus_b'):
      wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
      outputs = wx_plus_b
    else:
      outputs = activation_function(wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
  xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
  ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
layer1 = add_layer(1, xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(2, layer1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
  tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
  train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter('logs/', sess.graph)
  sess.run(init)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(x_data, y_data)
  plt.ion()
  plt.show()
  for i in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
      print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
      try:
        ax.lines.remove(ax.lines[0])
      except Exception:
        pass
      prediction_value = sess.run(prediction, feed_dict={xs: x_data})
      lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
      plt.pause(0.3)
      result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
      writer.add_summary(result, i)
      
