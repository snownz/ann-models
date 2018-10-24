import tensorflow as tf
import numpy as np

class NeuralNode_v0:    

    def __init__(self, lr, momentum, size, dropout, parent_node, name, act = tf.nn.relu ):
        
        self.lr = lr
        self.momentum = momentum
        self.dropout = dropout
        self.name = name 
        self.Y = tf.placeholder( tf.float32, [ None, size ], name = "Y_{}".format( name ) )

        if dropout > 0:
            self.layer = tf.layers.dropout( inputs = tf.layers.dense( inputs = parent_node, units = size, activation = act ), rate = dropout, name = name )
        else:
            self.layer = tf.layers.dense( inputs = parent_node, units = size, activation = act, name = name )

    def set_params(self, w, b):
        self.layer.setWeights( w )
        self.layer.setBiases( w )

    def get_params(self):
        w = self.layer.getWeights()
        b = self.layer.getBiases()

        return w, b

class MultitaskModel_v0:    
    
    def __init__(self, session, add_main = False, input_size = 0, size = 0, dropout = 0, name = "main", act = tf.nn.relu):                         
 
        self.tasks = []
        self.layers = []
        self.session = session
        self.input_size = input_size
        self.y_name = 0

        if add_main:

            self.X = tf.placeholder( tf.float32, [ None, input_size ], name = "X" )    
            self.main = NeuralNode_v0( lr = 0, momentum = 0, size = size, dropout = 0.5, parent_node = self.X, name = "main", act = act )
            self.layers.append( self.main )

    def add_layer(self, layer):
        self.layers.append( layer )

    def add_layers(self, layers):
        self.layers.extend( layers )

    def add_task(self, task):
        self.y_name += 1
        self.tasks.append( task )
        self.layers.append( task )
        self.Y = tf.placeholder( tf.float32, [ len( self.tasks ), None, 2 ], name = "Y_{}".format( self.y_name ) )

    def add_tasks(self, tasks):
        self.y_name += 1
        self.tasks.extend( tasks )
        self.layers.extend( tasks )
        self.Y = tf.placeholder( tf.float32, [ len( self.tasks ), None, 2 ], name = "Y_{}".format( self.y_name ) )

    def build_opmt(self, mode):
        
        # Y Axis 2 = layer
        # Y Axis 1 = samples
        # Y Axis 0 = tasks

        if mode == 'categorical':

            self.losses = tf.reduce_sum( 
                                tf.reduce_mean( 
                                    tf.negative( 
                                        tf.reduce_sum( 
                                            self.Y * tf.log( self.pd ), axis = 2 ) ), axis = 1 ) )

            self.accuracy = tf.reduce_mean( 
                                tf.cast( 
                                    tf.equal( 
                                        tf.argmax( self.Y, axis = 2 ), tf.argmax( self.pd, axis = 2 ) ), "float" ), axis = 1 )  
        
        else:

            self.losses = tf.reduce_sum( 
                                tf.reduce_mean( 
                                        tf.reduce_sum( 
                                             tf.pow( ( self.Y - self.pd ), 2 ), axis = 2 ), axis = 1 ) )

        
        self.optimizer_all = tf.train.MomentumOptimizer( learning_rate = 0.001, momentum = 0.1 ).minimize( self.losses )

    def build_feedforward(self):

        self.pd = [ x.layer for x in self.tasks ]

    def initilize_model(self, mode, varibales = False):

        self.build_feedforward()
        self.build_opmt( mode )

        if varibales:
            self.session.run( tf.global_variables_initializer() )
            self.session.run( tf.local_variables_initializer() )

    def train(self, input, target):
        
        _, loss, acc = self.session.run( [ self.optimizer_all, self.losses, self.accuracy ], { 'X:0' : input, 'Y_{}:0'.format( self.y_name ) : target } )
        return loss, acc

    def forward(self, input):

        return  self.session.run( self.pd, { 'X:0' : input } )

class NeuralNode_v1:    

    def __init__(self, lr, momentum, size, dropout, parent_node, name, mode = None, act = tf.nn.relu ):
        
        self.lr = lr
        self.momentum = momentum
        self.dropout = dropout
        self.name = name 
        self.Y = tf.placeholder( tf.float32, [ None, size ], name = "Y_{}".format( name ) )

        if dropout > 0:
            self.layer = tf.layers.dropout( inputs = tf.layers.dense( inputs = parent_node, units = size, activation = act ), rate = dropout, name = name )
        else:
            self.layer = tf.layers.dense( inputs = parent_node, units = size, activation = act, name = name )

        if mode == 'categorical':
            self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = self.layer, labels = self.Y ) )
            self.accuracy = tf.cast( tf.equal( tf.argmax( self.Y, 1 ), tf.argmax( self.layer, 1 ) ), "float" )

        if mode == 'regression':
            self.loss = tf.losses.huber_loss( labels = self.Y, predictions = self.layer )
            self.accuracy = tf.reduce_mean( 1 - tf.abs( self.Y - self.layer ), axis = 1 )

    def set_params(self, w, b):
        self.layer.setWeights( w )
        self.layer.setBiases( w )

    def get_params(self):
        w = self.layer.getWeights()
        b = self.layer.getBiases()

        return w, b

class MultitaskModel_v1:    
    
    def __init__(self, session, add_main = False, input_size = 0, size = 0, dropout = 0, name = "main", act = tf.nn.relu):                         
 
        self.tasks = []
        self.layers = []
        self.session = session

        if add_main:
            self.input_size = input_size
            self.X = tf.placeholder( tf.float32, [ None, input_size ], name = "X" )    
            self.main = NeuralNode_v1( lr = 0, momentum = 0, size = size, dropout = 0.5, parent_node = self.X, name = "main", act = act )
            self.layers.append( self.main )

    def add_layer(self, layer):
        self.layers.append( layer )

    def add_layers(self, layers):
        self.layers.extend( layers )

    def add_task(self, task):
        self.tasks.append( task )

    def add_tasks(self, tasks):
        self.tasks.extend( tasks )

    def build_opmt(self):
        
        self.losses = tf.reduce_sum( self.ls )
        self.accuracy = tf.reduce_mean( self.ac, axis = 1 )
        
        self.optimizer_all = tf.train.MomentumOptimizer( learning_rate = 0.001, momentum = 0.1 ).minimize( self.losses )

    def build_feedforward(self):

        self.pd = [ x.layer for x in self.tasks ]
        self.ls = [ x.loss for x in self.tasks ]
        self.ac = [ x.accuracy for x in self.tasks ]

    def initilize_model(self):

        self.build_feedforward()
        self.build_opmt()

        self.session.run( tf.global_variables_initializer() )
        self.session.run( tf.local_variables_initializer() )

    def train(self, input, target):
        
        feeds = {}
        feeds.update( { 'X:0': input } )
        for i, y in enumerate( self.tasks ):
            feeds.update( { 'Y_{}:0'.format( y.name ): target[ i ] } )

        _, loss, acc = self.session.run( [ self.optimizer_all, self.losses, self.accuracy ], feeds )

        return loss, acc

    def forward(self, input):
        
        return self.session.run( self.pd, { 'X:0' : input } )


#multilevel gradients
#var_list1 = [variables from first 5 layers]
#var_list2 = [the rest of variables]
#opt1 = tf.train.GradientDescentOptimizer(0.00001)
#opt2 = tf.train.GradientDescentOptimizer(0.0001)
#grads = tf.gradients(loss, var_list1 + var_list2)
#grads1 = grads[:len(var_list1)]
#grads2 = grads[len(var_list1):]
#tran_op1 = opt1.apply_gradients(zip(grads1, var_list1))
#train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
#train_op = tf.group(train_op1, train_op2)
#layers = [ x.layer for x in self.layers ] 
#grads = tf.gradients( self.losses, layers)
#opt = [ tf.train.MomentumOptimizer( learning_rate = x.lr , momentum = x.momentum ).apply_gradients( zip( grads[ i ], layers[ i ] ) ) for i, x in enumerate( self.layers ) ]