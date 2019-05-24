import tensorflow as tf
import numpy as np

from ann_utils.tf_helper import maxpool2d, avgpool2d
from ann_utils.tf_conv_layer import Conv2dLayer, SeparableConv2DLayer

'''
  Xception: Deep Learning with Depthwise Separable Convolutions
      Font: https://arxiv.org/pdf/1610.02357.pdf
'''
class Xception(object):

    def __init__( self,
                  name, scope,
                  dropout = 0.0, bn = False,
                  padding = "SAME",  
                  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), 
                  kernel_regularizer = None,
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu, 
                  trainable = True ):

        self.act = act

        #===========ENTRY FLOW==============
        # block1
        self.b1c1 = Conv2dLayer( 32, 3, 2, '{}_block1_conv1'.format( name ), scope,
                                 dropout, True, "VALID", kernel_initializer, kernel_regularizer, l1, l2, act, trainable ) 
        self.b1c2 = Conv2dLayer( 64, 3, 1, '{}_block1_conv2'.format( name ), scope,
                                 dropout, True, "VALID", kernel_initializer, kernel_regularizer, l1, l2, act, trainable ) 
        self.b1r1 = Conv2dLayer( 128, 1, 2, '{}_block1_res_conv'.format( name ), scope,
                                 dropout, bn, padding, kernel_initializer, kernel_regularizer, l1, l2, None, trainable ) 
        
        # block2
        self.b2c1 = SeparableConv2DLayer( 128, 3, 1, '{}_block2_dws_conv1'.format( name ), scope,
                                 dropout, True, padding, l1, l2, act, trainable ) 
        self.b2c2 = SeparableConv2DLayer( 128, 3, 1, '{}_block2_dws_conv2'.format( name ), scope,
                                 dropout, True, padding, l1, l2, act, trainable ) 
        self.b2r1 = Conv2dLayer( 256, 1, 2, '{}_block2_res_conv'.format( name ), scope,
                                 dropout, bn, padding, kernel_initializer, kernel_regularizer, l1, l2, None, trainable )
        
        # block3
        self.b3c1 = SeparableConv2DLayer( 256, 3, 1, '{}_block3_dws_conv1'.format( name ), scope,
                                 dropout, True, padding, l1, l2, act, trainable ) 
        self.b3c2 = SeparableConv2DLayer( 256, 3, 1, '{}_block3_dws_conv2'.format( name ), scope,
                                 dropout, True, padding, l1, l2, act, trainable ) 
        self.b3r1 = Conv2dLayer( 728, 1, 2, '{}_block3_res_conv'.format( name ), scope,
                                 dropout, bn, padding, kernel_initializer, kernel_regularizer, l1, l2, None, trainable )

        # block4
        self.b4c1 = SeparableConv2DLayer( 728, 3, 1, '{}_block4_dws_conv1'.format( name ), scope,
                                 dropout, True, padding, l1, l2, act, trainable ) 
        self.b4c2 = SeparableConv2DLayer( 728, 3, 1, '{}_block4_dws_conv2'.format( name ), scope,
                                 dropout, True, padding, l1, l2, act, trainable )

        #===========MIDDLE FLOW===============
        self.block = []
        for i in range(8):
            block_prefix = 'block%s_' % (str(i + 5))
            self.block.append( 
                [
                    SeparableConv2DLayer( 728, 3, 1, '{}_block_{}_dws_conv1'.format( name, block_prefix ), scope,
                                        dropout, True, padding, l1, l2, act, trainable ),
                    SeparableConv2DLayer( 728, 3, 1, '{}_block_{}_dws_conv2'.format( name, block_prefix ), scope,
                                      dropout, True, padding, l1, l2, act, trainable ),
                    SeparableConv2DLayer( 728, 3, 1, '{}_block_{}_dws_conv3'.format( name, block_prefix ), scope,
                                      dropout, True, padding, l1, l2, act, trainable )
                ]
            )
        
        #========EXIT FLOW============
        self.er0 = Conv2dLayer( 1024, 1, 2, '{}_block12_res_conv'.format( name ), scope,
                                dropout, bn, padding, kernel_initializer, kernel_regularizer, l1, l2, None, trainable )

        self.ec0 = SeparableConv2DLayer( 728, 3, 1, '{}_block13_dws_conv1'.format( name ), scope,
                                         dropout, True, padding, l1, l2, act, trainable )
        self.ec1 = SeparableConv2DLayer( 1024, 3, 1, '{}_block13_dws_conv2'.format( name ), scope,
                                         dropout, True, padding, l1, l2, act, trainable )
        
        self.ec2 = SeparableConv2DLayer( 1536, 3, 1, '{}_block14_dws_conv1'.format( name ), scope,
                                         dropout, True, padding, l1, l2, act, trainable )
        self.ec3 = SeparableConv2DLayer( 2048, 3, 1, '{}_block14_dws_conv2'.format( name ), scope,
                                         dropout, True, padding, l1, l2, act, trainable ) 
        
    def __call__(self, input, reuse=False, is_training=False):

        x = input

        #===========ENTRY FLOW==============
        # Block 1
        x = self.b1c1( x, reuse, is_training )
        x = self.b1c2( x, reuse, is_training )
        res = self.b1r1( x, reuse, is_training )
        
        # Block 2
        x = self.b2c1( x, reuse, is_training )
        x = self.b2c2( x, reuse, is_training )
        x = maxpool2d( x )
        x = tf.add( x, res )
        res = self.b2r1( x, reuse, is_training )
        
        # Block 3
        x = self.act( x )
        x = self.b3c1( x, reuse, is_training )
        x = self.b3c2( x, reuse, is_training )
        x = maxpool2d( x )
        x = tf.add( x, res )
        res = self.b3r1( x, reuse, is_training )
        
        # Block 4
        x = self.act( x )
        x = self.b4c1( x, reuse, is_training )
        x = self.b4c2( x, reuse, is_training )
        x = maxpool2d( x )
        x = tf.add( x, res )

        #===========MIDDLE FLOW===============
        for i in range(8):
            res = x
            for n in self.block[i]:
                x = self.act( x )
                x = n( x, reuse, is_training )
            x = tf.add( x, res )
        
        #========EXIT FLOW============
        res = self.er0( x, reuse, is_training )
        
        x = self.act( x )
        x = self.ec0( x, reuse, is_training )
        x = self.ec1( x, reuse, is_training )
        x = maxpool2d( x )
        x = tf.add( x, res )
        
        x = self.ec2( x, reuse, is_training )
        x = self.ec3( x, reuse, is_training )

        x = avgpool2d( x, 10 )

        if not reuse: self.layer = x
        
        print(x)            
        return x
        