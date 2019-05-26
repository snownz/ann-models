import tensorflow as tf
import numpy as np

from ann_utils.helper import dropout, bn

class NacCell(object):

    def __init__( self, 
                  output,  
                  name,
                  dropout = 0.0, 
                  l1 = 0.0, l2 = 0.0, 
                  act = None, 
                  trainable = True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.name = name 
        self.l1 = l1 
        self.l2 = l2 
        self.act = act 
        self.trainable = trainable

    def __call__(self, input, reuse=False, is_training=False): 

        # setup layer
        x = input
        input_size = input.shape[1].value 

        wt = tf.get_variable( "wt_{}".format( self.name ), 
                              [ input_size, self.output_size ], 
                              dtype = tf.float32, 
                              initializer = tf.truncated_normal_initializer( stddev = .01 ) 
                            )     

        mt = tf.get_variable( "mt_{}".format( self.name ), 
                              [ input_size, self.output_size ], 
                              dtype = tf.float32, 
                              initializer = tf.truncated_normal_initializer( stddev = .01 ) 
                            )

        w = tf.multiply( tf.tanh( wt ), tf.sigmoid( mt ) )

        x = tf.matmul( x, w )   
        
        # activation
        if not self.act is None:
            x = self.act( x )

        # setup dropout
        if self.dropout > 0 and is_training:
            x = tf.layers.dropout( inputs = x, rate = self.dropout )

        if not reuse: self.layer = x        
        
        print(x)            
        return x