import tensorflow as tf
from ann_utils.helper import binary_cross_entropy

def cross_entropy_cost(encoder, output, target, lr):

  a1 = 0.125 * tf.reduce_mean( binary_cross_entropy( target[ :, :, 0:3],   output[ :, :, 0:3]   ) )
  a2 = 0.125 * tf.reduce_mean( binary_cross_entropy( target[ :, :, 3:8],   output[ :, :, 3:8]   ) )
  a3 = 0.125 * tf.reduce_mean( binary_cross_entropy( target[ :, :, 8:18],  output[ :, :, 8:18]  ) )
  a4 = 0.125 * tf.reduce_mean( binary_cross_entropy( target[ :, :, 18:26], output[ :, :, 18:26] ) )
  a5 = 0.125 * tf.reduce_mean( binary_cross_entropy( target[ :, :, 26:28], output[ :, :, 26:28] ) )
  a7 = 0.125 * tf.reduce_mean( binary_cross_entropy( target[ :, :, 30:33], output[ :, :, 30:33] ) )

  a6 = 0.125 * tf.losses.huber_loss( target[ :, :, 28:30], output[ :, :, 28:30] )
  a8 = 0.025 * tf.losses.huber_loss( target[ :, :, 33:36], output[ :, :, 33:36] )

  correct_pred1 = tf.equal( tf.argmax( output[ :, :, 0:3],   2 ), tf.argmax( target[ :, :, 0:3],   2 ) )
  correct_pred2 = tf.equal( tf.argmax( output[ :, :, 3:8],   2 ), tf.argmax( target[ :, :, 3:8],   2 ) )
  correct_pred3 = tf.equal( tf.argmax( output[ :, :, 8:18],  2 ), tf.argmax( target[ :, :, 8:18],  2 ) )
  correct_pred4 = tf.equal( tf.argmax( output[ :, :, 18:26], 2 ), tf.argmax( target[ :, :, 18:26], 2 ) )
  correct_pred5 = tf.equal( tf.argmax( output[ :, :, 26:28], 2 ), tf.argmax( target[ :, :, 26:28], 2 ) )
  correct_pred6 = tf.equal( tf.argmax( output[ :, :, 30:33], 2 ), tf.argmax( target[ :, :, 30:33], 2 ) )

  accuracy1 = 0.166666 * tf.reduce_mean( tf.cast( correct_pred1, tf.float32 ) )
  accuracy2 = 0.166666 * tf.reduce_mean( tf.cast( correct_pred2, tf.float32 ) )
  accuracy3 = 0.166666 * tf.reduce_mean( tf.cast( correct_pred3, tf.float32 ) )
  accuracy4 = 0.166666 * tf.reduce_mean( tf.cast( correct_pred4, tf.float32 ) )
  accuracy5 = 0.166666 * tf.reduce_mean( tf.cast( correct_pred5, tf.float32 ) )
  accuracy6 = 0.166666 * tf.reduce_mean( tf.cast( correct_pred6, tf.float32 ) )

  loss = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
  accuracy = accuracy1 + accuracy2 + accuracy3 + accuracy4 + accuracy5 + accuracy6
  
  op = tf.train.RMSPropOptimizer( lr ).minimize( loss, var_list = encoder.get_all_params()[1] )
  
  return [ op, loss, accuracy ]