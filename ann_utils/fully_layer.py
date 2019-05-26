import tensorflow as tf
import numpy as np
from ai_utils.helper import dropout

class FullyLayer(object):

    def __init__(self, size, name, dropout = 0.0, l1 = 0.0, l2 = 0.0, act = tf.nn.relu, trainable = True, kernel_regularizer = None):
        
        self.size = size 
        self.dropout = dropout 
        self.name = name 
        self.l1 = l1 
        self.l2 = l2 
        self.act = act 
        self.trainable = trainable
        self.kernel_regularizer = kernel_regularizer

    def __call__(self, input, reuse=False, is_training=False): 
        
        # setup layer params
        x = input        
        x = tf.layers.dense( 
                            inputs = x, 
                            units = self.size, 
                            activation = self.act, 
                            trainable = self.trainable,
                            kernel_regularizer = self.kernel_regularizer,
                            name = self.name
                        )
        
        # setup dropout
        if self.dropout > 0 and is_training:
            x = dropout( x, self.dropout )

        if not reuse: self.layer = x
        
        print(x)            
        return x
