import tensorflow as tf
import numpy as np

from ai_utils.helper import maxpool2d, avgpool2d, dropout, bn

class Conv2DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dropout = 0.0, bn = False,
                  padding = "SAME",  
                  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), 
                  kernel_regularizer = None,
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu, 
                  trainable = True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.bn = bn 
        self.name = name 
        self.l1 = l1 
        self.l2 = l2 
        self.act = act 
        self.trainable = trainable
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def __call__(self, x, reuse=False, is_training=False): 

        # setup layer
        x = tf.layers.conv2d( x,
                              filters            = self.output_size,
                              kernel_initializer = self.kernel_initializer,
                              kernel_regularizer = self.kernel_regularizer,
                              bias_initializer   = tf.zeros_initializer(),
                              kernel_size        = [ self.kernel, self.kernel ] if type(self.kernel) is type(int) else self.kernel, 
                              strides            = [ self.stride, self.stride ],
                              padding            = self.padding,
                              trainable          = self.trainable,
                              name               = self.name
                             )

        # batch normalization
        if self.bn:
            x = bn( x )

        # activation
        if not self.act is None:
            x = self.act( x )

        # setup dropout
        if self.dropout > 0 and is_training:
            x = dropout( x, self.dropout )

        if not reuse: self.layer = x    
        
        print(x)            
        return x

class Deconv2DLayer( Conv2DLayer ):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dropout = 0.0, bn = False,
                  padding = "SAME",  
                  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), 
                  kernel_regularizer = None,
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu, 
                  trainable = True
                ):

        super().__init__( output, kernel, stride, name,
                          dropout, bn, padding, kernel_initializer, kernel_regularizer,
                          l1, l2, act, trainable )

    def __call__(self, input, reuse=False, is_training=False): 

        # setup layer
        x = input

        x = tf.layers.conv2d_transpose( x,
                                        filters            = self.output_size,
                                        kernel_initializer = self.kernel_initializer,
                                        kernel_regularizer = self.kernel_regularizer,
                                        bias_initializer   = tf.zeros_initializer(),
                                        kernel_size        = [ self.kernel, self.kernel ] if type(self.kernel) is type(int) else self.kernel, 
                                        strides            = [ self.stride, self.stride ],
                                        padding            = self.padding,
                                        trainable          = self.trainable,
                                        name               = self.name
                                       )
        # batch normalization
        if self.bn:
            x = bn( x )
        
        # activation
        if not self.act is None:
            x = self.act( x )

        # setup dropout
        if self.dropout > 0 and is_training:
            x = dropout( x, self.dropout )
        
        if not reuse: self.layer = x        
        
        print(x)            
        return x

class SeparableConv2DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dropout = 0.0, bn = False,
                  padding = "SAME",  
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu, 
                  trainable = True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.bn = bn 
        self.name = name 
        self.l1 = l1 
        self.l2 = l2 
        self.act = act 
        self.trainable = trainable

    def __call__(self, input, reuse=False, is_training=False): 

        # setup layer
        x = input
        x = tf.layers.separable_conv2d( x,
                                        filters            = self.output_size,
                                        bias_initializer   = tf.zeros_initializer(),
                                        kernel_size        = [ self.kernel, self.kernel ] if type(self.kernel) is type(int) else self.kernel, 
                                        strides            = [ self.stride, self.stride ],
                                        padding            = self.padding,
                                        trainable          = self.trainable,
                                        name               = self.name
                                      )

        # batch normalization
        if self.bn:
            x = bn( x )

        # activation
        if not self.act is None:
            x = self.act( x )

        # setup dropout
        if self.dropout > 0 and is_training:
            x = dropout( x, self.dropout )

        if not reuse: self.layer = x    
        
        print(x)            
        return x

class Conv1DSequenceLayer(object):

    def __init__( self, 
                  output, 
                  name, 
                  w_init_stdev=0.02, 
                  act=None
                ):

        self.nf = output 
        self.name = name 
        self.act = act 
        self.w_init_stdev = w_init_stdev

    def __call__(self, x, mult=1, reuse=False, is_training=False): 

        if self.nf is None:
            self.nf = x.shape[-1].value * mult

        *start, nx = shape_list( x )
        w = tf.get_variable( 'conv_1D_w_{}'.format( self.name ), [ 1, nx, self.nf ], initializer = tf.random_normal_initializer( stddev = self.w_init_stdev ), trainable = is_training )
        b = tf.get_variable( 'conv_1D_b_{}'.format( self.name ), [ self.nf ], initializer = tf.constant_initializer(0), trainable = is_training )
        x = tf.reshape( tf.matmul( tf.reshape( x, [ -1, nx ] ), tf.reshape( w, [ -1, self.nf ] ) ) + b, start + [ self.nf ] )

        if not self.act is None:
            x = self.act( x ) 

        if not reuse: self.layer = x        
        
        print(x)            
        return x

class Conv1DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name, 
                  dropout = 0.0,
                  padding = "SAME",  
                  kernel_initializer = tf.contrib.layers.xavier_initializer(), 
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu, 
                  trainable = True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.name = name 
        self.l1 = l1 
        self.l2 = l2 
        self.act = act 
        self.trainable = trainable
        self.kernel_initializer = kernel_initializer

    def __call__(self, input, reuse=False, is_training=False): 

        # setup layer
        x = input
        x = tf.layers.conv1d( x,
                              filters            = self.output_size,
                              kernel_initializer = self.kernel_initializer,
                              bias_initializer   = tf.zeros_initializer(),
                              kernel_size        = self.kernel, 
                              strides            = self.stride,
                              padding            = self.padding,
                              trainable          = self.trainable,
                              name               = self.name
                             )

        # activation
        if not self.act is None:
            x = self.act( x )

        # setup dropout
        if self.dropout > 0 and is_training:
            x = tf.layers.dropout( inputs = x, rate = self.dropout )

        if not reuse: self.layer = x        
        
        print(x)            
        return x