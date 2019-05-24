import tensorflow as tf
from ann_utils.tf_helper import flatten

def soft_dice_loss(target, prediction, epsilon=1e-6): 

    target = tf.round( tf.layers.flatten( target, "loss" ) / 255.0 )
    prediction = tf.layers.flatten( prediction, "loss" ) / 255.0

    numerator = 2. * tf.reduce_sum( prediction * target, axis = 1 )
    denominator = tf.reduce_sum( tf.square( prediction ) + tf.square( target ), axis = 1 ) 
    
    return 1 - tf.reduce_mean( numerator / ( denominator + epsilon ) ) # average over classes and batch

def single_black_loss(target, prediction, th=200): 

    # inverse image
    i1i = 255.0 - target
    i2i = 255.0 - prediction

    # binarize
    i1t = tf.cast( i1i > th, tf.float32 )
    i2t = tf.cast( i2i > th, tf.float32 )

    # compare
    c = i2t - i1t
    d = tf.cast( c < 0, tf.float32 )
    e = tf.cast( c > 0, tf.float32 ) * 0.5
    diff_error = d + e

    # reduce error
    return flatten( diff_error )