import tensorflow as tf
import numpy as np

# =============================================== costs =============================================== 

def binary_cross_entropy(target, predict, eps=1e-12):
    return ( -( target * tf.log( predict + eps ) + ( 1. - target ) * tf.log( 1. - predict + eps ) ) )

def huber_loss(x, delta=1.0):    
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

def l2(scale):
    return tf.contrib.layers.l2_regularizer( scale = scale )

# =============================================== optmizer operations =============================================== 

def flatgrad(loss, var_list, clip_norm=None):
    
    grads = tf.gradients( loss, var_list )
    
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])

# =============================================== operations =============================================== 

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

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

# =============================================== non-lieanr functions =============================================== 

def gelu(x):
    x = 0.5 * x * ( 1 + tf.tanh( np.sqrt( 2 / np.pi ) * ( x + 0.044715 * tf.pow( x, 3 ) ) ) )
    #print(x)
    return x

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

# =============================================== layers =============================================== 

def bn(x, center=True, scale=True, decay=0.9, updates_collections=None):
    x = tf.contrib.layers.batch_norm( x, center = center, scale = scale, decay = decay, updates_collections = updates_collections )
    #print(x)
    return x
