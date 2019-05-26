import tensorflow as tf
from tensorflow.python.ops import array_ops

from ann_utils.conv_layer import Conv2DLayer, SeparableConv2DLayer
from ann_utils.som_layer import SOMLayer
from ann_utils.fully_layer import FullyLayer
from ann_utils.helper import flatten, l2
from ann_utils.nalu_cell import NaluCell
from ann_utils.nac_cell import NacCell

class MemoryCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, name, memory, state_is_tuple=False, act=tf.nn.relu):

        tf.nn.rnn_cell.RNNCell.__init__(self)

        self.is_training = True

        self._num_units = num_units
        self.memory = memory
        self._state_is_tuple = state_is_tuple

        self.f = FullyLayer( self._num_units, "forget_{}".format( name ), act = tf.nn.sigmoid ) # forget memory
        self.i = FullyLayer( self._num_units, "ignore_{}".format( name ), act = tf.nn.sigmoid ) # ignore prediction
        self.o = FullyLayer( self._num_units, "output_{}".format( name ), act = tf.nn.sigmoid ) # output prediction
        self.c = NaluCell( self._num_units, "cell_{}".format( name ), act = act ) # add memory

        self.fg = FullyLayer( self._num_units, "fg_{}".format( name ), act = tf.nn.sigmoid )
        self.ig = FullyLayer( self._num_units, "ig_{}".format( name ), act = tf.nn.sigmoid )        
        self.mg = FullyLayer( self._num_units, "mg_{}".format( name ), act = tf.nn.sigmoid )        
        
        self.som = memory
        
        self.c1 = Conv2DLayer( 8,  5, 2, "c1_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )
        self.c2 = Conv2DLayer( 16, 3, 2, "c2_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )
        self.c3 = Conv2DLayer( 32, 2, 2, "c3_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )        
        self.c4 = Conv2DLayer( 64, 1, 2, "c4_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )
        self.nm = NaluCell( self._num_units, "nm_{}".format( name ), act = act )        

    @property
    def state_size(self):
        return self._num_units * 3

    @property
    def output_size(self):
        return self._num_units

    def set_values(self, is_training):
        self.is_training = is_training

    def __encode(self, x):
        
        x = self.c1( x, is_training = self.is_training )
        x = self.c2( x, is_training = self.is_training )
        x = self.c3( x, is_training = self.is_training )
        x = self.c4( x, is_training = self.is_training )
        x = flatten( x )    

        return self.nm( x, is_training = self.is_training ) 

    def _retrive_memory(self, x, h):

        mems = self.som( x, h )        
        mem = self.__encode( mems )

        return mem

    def call(self, x, state):
        
        if type(state) is tuple:
            mem     = state[0]
            h       = state[1]
            c_prior = state[2]
        else:
            mem     = array_ops.slice( state, [ 0, 0                   ], [ -1, self._num_units ] )
            h       = array_ops.slice( state, [ 0, self._num_units     ], [ -1, self._num_units ] )
            c_prior = array_ops.slice( state, [ 0, self._num_units * 2 ], [ -1, self._num_units ] )

        # gates inputs
        mx = tf.concat( [ mem, x ], axis = 1 )
        hx = tf.concat( [ h, x ], axis = 1 )

        # gates for short memory
        f = self.f( hx, is_training = self.is_training )
        i = self.i( hx, is_training = self.is_training )
        c = self.c( hx, is_training = self.is_training )

        # compute short memory
        _c = ( f * c_prior ) + ( ( ( 1.0 - f ) * i ) * c )       

        # gates for long memory
        fg = self.fg( mx, is_training = self.is_training )
        ig = self.ig( mx, is_training = self.is_training )

        # ignore gate should consider only empts positions by forget gate
        # compute new long memory
        new_m = ( fg * mem ) + ( ( ( 1.0 - fg ) * ig ) * self._retrive_memory( x, h ) )
        # new_m = tf.zeros_like( _c )

        hxm = tf.concat( [ hx, new_m ], axis = 1 )

        # new hidden state
        o = self.o( hxm, is_training = self.is_training )
        _h = ( o * tf.nn.tanh( _c ) ) + new_m

        # compute long memory gate
        mg = self.mg( tf.abs( _h - h ), is_training = self.is_training )
        # mg = self.mg( o, is_training = self.is_training )

        # get hidden info to update memory and next time
        mem_update = mg * _h
        new_h = ( 1.0 - mg ) * _h

        # pass out to non-linear function
        out = tf.nn.tanh( _h )

        return ( out, mem_update ), array_ops.concat( [ new_m, new_h, _c ], 1 )