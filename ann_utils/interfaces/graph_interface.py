import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf

from ai_utils.manager import tf_load, tf_save

class GraphInterface:

    def __init__(self, folder, scope, session):
        self.folder = folder
        self.scope  = scope
        self.sess   = session

    def get_all_params(self):
        params = [ x for x in tf.trainable_variables() if self.scope in x.name ]
        params_names = [ x.name for x in tf.trainable_variables() if self.scope in x.name ]
        return params_names, params

    def get_all_params_values(self):
        params, params_names = self.get_all_params() 
        return params_names, self.sess( params )

    def save_model(self):
        tf_save( self.folder, self.get_all_params(), self.scope, self.sess )

    def load_model(self):
        tf_load( self.folder, self.get_all_params(), self.scope, self.sess)