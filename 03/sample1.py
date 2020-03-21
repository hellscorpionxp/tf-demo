'''
Created on 2020年3月19日

@author: tony
'''
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

state = tf.Variable(0, name='counter')
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))