import os, warnings
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

from sklearn.exceptions import DataConversionWarning

class TfSess(object):

    def __init__(self, percent=False, remote=None, gpu=False, disable_warnings=True):

        self.percent = percent
        self.remote = remote
        self.gpu = gpu
        self.writer = tf.summary.FileWriter( '/tmp/tensorflow/rnn_words' )
        self.session = None
        self.reset()        

    def reset(self):

        if not self.session is None:
            self.session.close()

        if not self.gpu:
            config = tf.ConfigProto( device_count = { 'GPU': 0 } )
        else:
            config = tf.ConfigProto()
        
        if self.percent:
            config.gpu_options.per_process_gpu_memory_fraction=0.3
                    
        if self.remote is None:
            self.session = tf.InteractiveSession(config=config)
        else:
            self.session = tf.InteractiveSession(self.remote, config=config)

        if self.gpu:
            device_name = tf.test.gpu_device_name()
            print('Found GPU at: {}'.format(device_name))

    def tensorboard_graph(self):
        self.writer.add_graph( self.session.graph )

    def get_session(self):
        return self.session

    def __call__(self, tensor, inputs=None):
        if inputs is None:
            return self.session.run( tensor )
        else:
            return self.session.run( tensor, feed_dict = inputs )