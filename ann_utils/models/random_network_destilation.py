import sys
sys.path.append('../../')

import tensorflow as tf

from ai_utils.models.double_deep_qlearning import DDQL

from ai_utils.interfaces.graph_interface import GraphInterface 
from ai_utils.fully_layer import FullyLayer 
from ai_utils.helper import gelu 

class DDQLRND( GraphInterface ):

    def __init__(self, s_encode_size, h_size, n_a, folder, scope, session):

        self.scope = scope

        self.r_n = FullyLayer( s_encode_size, '{}_random_net'.format( scope ), act = None )
        
        self.p_h = FullyLayer( h_size, '{}_p_h'.format( scope ), act = gelu )
        self.p_o = FullyLayer( s_encode_size, '{}_p_o'.format( scope ), act = None )
        
        self.ddqn = DDQL( s_encode_size, h_size, n_a, folder, scope, session )
        

    def build_training_graph(self, n_s, gamma, lr, soft_update=0.9):

        with tf.variable_scope( self.scope, reuse = tf.AUTO_REUSE ):

            st, ac, er, s_ = self.ddqn._build_inputs( n_s )

            pred_train, dqn_train, q, dqn_loss, soft_update_op = self._build_nets(  st, ac, er, s_, gamma, lr, soft_update, True )
         
            return st, ac, er, s_, pred_train, dqn_train, q, soft_update_op, dqn_loss
    
    def _build_nets(self, st, ac, er, s_, gamma, lr, soft_update, is_training):
        
        # fixed random net
        with tf.variable_scope("random_net"):
            rand_encode_s_ = self.r_n( s_, is_training = False ) # s_encode_size

        # predictor
        ri, pred_train = self._build_predictor( s_, rand_encode_s_, lr, is_training )

        # normal RL model
        q, q_, soft_update_op = self.ddqn._build_net(st, ac, er, s_, gamma, soft_update, True )

        # ddqn loss
        dqn_loss, dqn_train = self._build_ddqn_optmizer( er, ri, gamma, q_, ac, q, lr )

        return pred_train, dqn_train, q, dqn_loss, soft_update_op

    def _build_predictor(self, s_, rand_encode_s_, lr, is_training):

        with tf.variable_scope("predictor"):

            x = self.p_h( s_, is_training = is_training ) # 128 relu
            x = self.p_o( x,  is_training = is_training ) # s_encode_size

        with tf.name_scope("int_r"):
            ri = tf.reduce_sum( tf.square( rand_encode_s_ - x ), axis = 1 )  # intrinsic reward
        
        train_op = tf.train.MomentumOptimizer( lr, 0.9, name = "predictor_opt" ).minimize(
            tf.reduce_mean( ri ), var_list = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, "{}/predictor".format( self.scope ) ) )

        return ri, train_op

    def _build_ddqn_optmizer(self, re, ri, gamma, q_, ac, q, lr):
        
        with tf.variable_scope('q_target'):
            q_target = re + ri + gamma * tf.reduce_max( q_, axis = 1, name = "Qmax_s_" )

        with tf.variable_scope('q_wrt_a'):
            a_indices = tf.stack( [ tf.range( tf.shape( ac )[0], dtype = tf.int32), ac ], axis = 1 )
            q_wrt_a = tf.gather_nd( params = q, indices = a_indices )

        loss = tf.losses.mean_squared_error( labels = q_target, predictions = q_wrt_a ) # dqn error

        train_op = tf.train.MomentumOptimizer( lr, 0.9, name = "dqn_opt" ).minimize(
            loss, var_list = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, "{}/eval_net".format( self.scope ) ) )
        
        return loss, train_op
