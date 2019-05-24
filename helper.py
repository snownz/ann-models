import tensorflow as tf
import numpy as np

def binary_cross_entropy(target, predict, eps=1e-12):
    return ( -( target * tf.log( predict + eps ) + ( 1. - target ) * tf.log( 1. - predict + eps ) ) )

def l2(scale):
    return tf.contrib.layers.l2_regularizer( scale = scale )

def flatten(x):
    x = tf.layers.flatten( x )
    #print(x)
    return x        

def dropout(x, dp=0.5):
    x = tf.layers.dropout( x, dp )
    #print(x)
    return x

def concat(x, axis):
    x = tf.concat( x, axis = axis )
    #print(x)
    return x

def maxpool2d(x, k=2, s=1):
    ksize = [s, k, k, s]
    x = tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME')
    #print(x)
    return x

def avgpool2d(x, k=2, s=1):
    ksize = [s, k, k, s]
    x = tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='SAME')
    #print(x)
    return x

def gelu(x):
    x = 0.5 * x * ( 1 + tf.tanh( np.sqrt( 2 / np.pi ) * ( x + 0.044715 * tf.pow( x, 3 ) ) ) )
    #print(x)
    return x

def bn(x, center=True, scale=True, decay=0.9, updates_collections=None):
    x = tf.contrib.layers.batch_norm( x, center = True, scale = True, decay = 0.9, updates_collections = None )
    #print(x)
    return x