import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf

from ai_utils.manager import tf_load, tf_save

class GraphInterface:

    def __init__(self, folder, scope, session, input):
        self.folder = folder
        self.output = None
        self.scope  = scope
        self.sess   = session
        self.input  = input

    def get_all_params(self):
        params = [ x for x in tf.trainable_variables() if self.scope in x.name ]
        params_names = [ x.name for x in tf.trainable_variables() if self.scope in x.name ]

        return params_names, params

    def get_all_params_values(self):
        params = [ x for x in tf.trainable_variables() if self.scope in x.name ]
        params_names = [ x.name for x in tf.trainable_variables() if self.scope in x.name ]
        
        return params_names, self.sess.get_session().run( params )

    def save_model(self):
        tf_save( self.folder, self.get_all_params(), self.scope, self.sess )

    def load_model(self):
        tf_load( self.folder, self.get_all_params(), self.scope, self.sess)
    
    def build_graph(self, is_training=False, reuse=False, input=None):       
        
        if not input is None: 
            x = input
        elif not self.input is None: 
            x = self.input
        else: 
            self._create_input()
            x = self.input
                    
        x = self._build_graph( is_training, reuse, x )            
        
        if not reuse: self.output = x
        return x
    
    def clone_and_build(self, new_name=None): 
        pass

    def _create_input(self):
        pass

    def _build_graph(self, is_training, reuse, x):
        return tf.no_op()

    def _set_input(self, input):
        self.input = input