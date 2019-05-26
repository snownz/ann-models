import sys
sys.path.append('../../')

import tensorflow as tf

from ai_utils.interfaces.graph_interface import GraphInterface 
from ai_utils.fully_layer import FullyLayer 
from ai_utils.helper import gelu 

class DDQL( GraphInterface ):

    def __init__(self, s_encode_size, h_size, n_a, folder, scope, session):

        self.scope = scope

        self.r_n = FullyLayer( s_encode_size, '{}_random_net'.format( scope ), act = None )
        
        self.p_h = FullyLayer( h_size, '{}_p_h'.format( scope ), act = gelu )
        self.p_o = FullyLayer( s_encode_size, '{}_p_o'.format( scope ), act = None )
        
        self.e_h = FullyLayer( h_size, '{}_e_h'.format( scope ), act = gelu )
        self.e_o = FullyLayer( n_a, '{}_e_o'.format( scope ), act = None )

        self.te_h = FullyLayer( h_size, '{}_te_h'.format( scope ), act = gelu )
        self.te_o = FullyLayer( n_a, '{}_te_o'.format( scope ), act = None )
        

    def build_training_graph(self, n_s, gamma, lr, soft_update=0.9):

        with tf.variable_scope( self.scope, reuse = tf.AUTO_REUSE ):

            # build inputs
            st, ac, er, s_ = self._build_inputs( n_s )
            
            # build network operations
            q, q_, soft_update_op = self._build_net(st, ac, er, s_, gamma, soft_update, True )
            
            # build optmizer
            dqn_loss, dqn_train = self._build_optmizer( er, gamma, q_, ac, q, lr )

            return st, ac, er, s_, dqn_train, q, soft_update_op, dqn_loss

    def _build_inputs(self, n_s):

        st = tf.placeholder( tf.float32, [None, n_s], name="s"     )  # input State
        ac = tf.placeholder( tf.int32,   [None, ],    name="a"     )  # input Action
        er = tf.placeholder( tf.float32, [None, ],    name="ext_r" )  # extrinsic reward
        s_ = tf.placeholder( tf.float32, [None, n_s], name="s_"    )  # input Next State

        return st, ac, er, s_   
    
    def _build_net(self, s, a, re, s_, gamma, soft_update,  is_training):

        with tf.variable_scope('eval_net'):
            x = self.e_h( s, is_training = is_training )
            q = self.e_o( x, is_training = is_training ) 

        with tf.variable_scope('target_net'):
            x_ = self.te_h( s_, is_training = is_training ) 
            q_ = self.te_o( x_, is_training = is_training )

        t_params = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope = '{}/target_net'.format( self.scope ) )
        e_params = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope = '{}/eval_net'.format( self.scope   ) )

        with tf.variable_scope('soft_update'):
            soft_update_op = [ tf.assign( t, ( soft_update * e ) + ( ( 1 - soft_update ) * t ) ) for t, e in zip( t_params, e_params ) ]

        return q, q_, soft_update_op

    def _build_optmizer(self, re, gamma, q_, ac, q, lr):

        with tf.variable_scope('q_target'):
            q_target = re + gamma * tf.reduce_max( q_, axis = 1, name = "Qmax_s_" )

        with tf.variable_scope('q_wrt_a'):
            a_indices = tf.stack( [ tf.range( tf.shape( ac )[0], dtype = tf.int32), ac ], axis = 1 )
            q_wrt_a = tf.gather_nd( params = q, indices = a_indices )

        loss = tf.losses.mean_squared_error( labels = q_target, predictions = q_wrt_a )

        train_op = tf.train.MomentumOptimizer( lr, 0.9, name = "dqn_opt" ).minimize(
            loss, var_list = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, "{}/eval_net".format( self.scope ) ) )

        return loss, train_op
