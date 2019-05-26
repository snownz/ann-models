import tensorflow as tf
import numpy as np

from ann_utils.helper import flatten, avgpool2d
from ann_utils.som_layer import SOMLayer

class StackedMemoryBlock(object):

    def __init__( self,
                  name,
                  m, n,
                  epoch,
                  lr,
                  nr,
                  stack,
                  act=None,
                  polling=None
                ):

        self.blocks = [ MemoryBlock( "{}_{}_mem_block".format( i, name ), 
                        m, n, epoch, lr, nr, act, polling )                         
                        for i in range( stack ) ]

    def __call__(self, x, h):

        mems = None
        for m in self.blocks:
            
            # retive memory information
            xi, hi, mi = m( x, h, is_training = False )
            fmi = flatten( mi )
            
            # store main memory
            if mems is None:
                mems = mi
            else:
                mems = tf.concat( [ mems, mi ], axis = 3 )

            # create input for next layer
            x = tf.concat( [ xi, fmi ], axis = 1 )
            h = tf.concat( [ hi, fmi ], axis = 1 )
        
        return mems

    def update_memory(self, x, h):

        update = []
        for m in self.blocks:

            # retive memory information
            xi, hi, mi, up = m( x, h, is_training = True )
            fmi = flatten( mi )

            # create input for next layer
            x = tf.concat( [ xi, fmi ], axis = 1 )
            h = tf.concat( [ hi, fmi ], axis = 1 ) 
            
            update.extend( up ) 

        return update     

class MemoryBlock(object):

    def __init__( self,
                  name,
                  m, n,
                  epoch,
                  lr,
                  nr,
                  act=None,
                  polling=None
                ):

        self.name = name 
        self.m = m 
        self.n = n 
        self.epoch = epoch
        self.lr = lr 
        self.nr = nr
        self.act = act
        self.polling = polling

        self.blockx = SOMLayer( "{}_block_x".format( name ), int( m / 2 ), int( n / 2 ), epoch, lr, nr, 0.05, act )
        self.blockh = SOMLayer( "{}_block_h".format( name ), int( m / 2 ), int( n / 2 ), epoch, lr, nr, 0.05, act )
        self.blockm = SOMLayer( "{}_block_m".format( name ), m, n, epoch, lr, nr, 0.05, act )

    def __call__(self, x, h, is_training=False):

        x, xu = self.blockx( x, is_training = is_training )
        h, hu = self.blockh( h, is_training = is_training )

        if not self.polling is None:
            x = avgpool2d( x, self.polling )
            h = avgpool2d( h, self.polling )

        x = flatten( x )
        h = flatten( h )
        xh = tf.concat( [ x, h ], axis  = 1 )

        m, mu = self.blockm( xh, is_training = is_training )

        # if not self.polling is None:
        #     m = avgpool2d( m, self.polling )
                
        print(m)               
        if is_training:
            return x, h, m, ( *xu, *hu, *mu )
        return x, h, m