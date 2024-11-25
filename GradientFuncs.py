import tensorflow as tf
import numpy as np

def kernelize(a):
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

# 2D convolution
def conv2D(x, k):
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

# gradient to get vertical lines
def gradientx(x):
  gx = kernelize([[-1.,0., 1.],
                            [-1.,0., 1.],
                            [-1.,0., 1.]])
  return conv2D(x, gx)

#gradient to get horizontal lines
def gradienty(y):
  gy = kernelize([[-1., -1, -1],
                            [0.,   0,  0], 
                            [1.,   1,  1]])
  return conv2D(y, gy)

