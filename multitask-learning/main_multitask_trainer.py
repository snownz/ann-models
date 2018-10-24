import numpy as np
import tensorflow as tf
from model.neural import *
from random import *
import matplotlib.pyplot as plt

ses = tf.Session()

X = [ ]
#Y = [ [] ]
Y = [ [],[] ]
Y = [ [],[],[] ]

for x in range( 100 ):
    X.append( np.array( [ uniform( 0.1, 0.9 ), 0, 0 ] ) )
    Y[0].append( np.array( [ 1, 0 ] ) )
    Y[1].append( np.array( [ 0, 1 ] ) )
    Y[2].append( np.array( [ 0, 1 ] ) )

for x in range( 50 ):
    X.append( np.array( [ 0, uniform( 0.1, 0.9 ), 0 ] ) )
    Y[1].append( np.array( [ 1, 0 ] ) )
    Y[0].append( np.array( [ 0, 1 ] ) )
    Y[2].append( np.array( [ 0, 1 ] ) )

for x in range( 10 ):
    X.append( np.array( [ 0, 0, uniform( 0.1, 0.9 ) ] ) )
    Y[2].append( np.array( [ 1, 0 ] ) )
    Y[1].append( np.array( [ 0, 1 ] ) )
    Y[0].append( np.array( [ 0, 1 ] ) )

mode = 0

if mode == 0:

    # Create a multitask Model
    model = MultitaskModel_v0( session = ses, add_main = True, input_size = 3, size = 10, dropout = 0.5 )

    # Define Shared Layers 
    c = NeuralNode_v0( lr = 0, momentum = 0, size = 7, dropout = 0.5, parent_node = model.main.layer, name = "c" )
    f = NeuralNode_v0( lr = 0, momentum = 0, size = 5, dropout = 0.5, parent_node = c.layer, name = "f" )

    # Add layers to Model
    model.add_layers( [ c, f ] )

    # Define Specific task Layers
    t0_h = NeuralNode_v0( lr = 0, momentum = 0, size = 3, dropout = 0, parent_node = f.layer, name = "task0_h" )
    t0 = NeuralNode_v0( lr = 0, momentum = 0, size = 2, dropout = 0, parent_node = t0_h.layer, name = "task0", act = tf.nn.softmax )

    t1_h = NeuralNode_v0( lr = 0, momentum = 0, size = 3, dropout = 0, parent_node = f.layer, name = "task1_h" )
    t1 = NeuralNode_v0( lr = 0, momentum = 0, size = 2, dropout = 0, parent_node = t1_h.layer, name = "task1", act = tf.nn.softmax )

    t2_h = NeuralNode_v0( lr = 0, momentum = 0, size = 3, dropout = 0, parent_node = f.layer, name = "task2_h" )
    t2 = NeuralNode_v0( lr = 0, momentum = 0, size = 2, dropout = 0, parent_node = t2_h.layer, name = "task2", act = tf.nn.softmax )

    # Add tasks to Model
    model.add_tasks( [ t0, t1, t2 ] )

    # Initialize model
    model.initilize_model( 'categorical', True )

    # Train
    batch_size = 10
    step = len( X ) / batch_size

    epoch = 10000000000

    acc = []
    loss = [ ]
    while len(loss) == 0 or loss[ -1 ] > 0.1:

        ac = []
        ls =[]
        for b in range( batch_size ):        
            
            init = int( b * step )
            end = int( init + step ) 

            l, a = model.train( 
                np.asarray( X [ init : end ] ), 
                np.asarray( [ x[ init : end ] for x in Y ] ) )

            ac.append( a )
            ls.append( l )

        acc.append( np.average( ac, axis=0 ) )
        loss.append( np.average( ls ) )
        
        print("Loss: {}\nAccuracy: {}\n\n".format( np.average( ls ), np.average( ac, axis=0 ) ))

else:

    # Create a multitask Model
    model = MultitaskModel_v1( session = ses, add_main = True, input_size = 3, size = 10, dropout = 0.5 )

    # Define Shared Layers 
    c = NeuralNode_v1( lr = 0, momentum = 0, size = 7, dropout = 0.5, parent_node = model.main.layer, name = "c" )
    f = NeuralNode_v1( lr = 0, momentum = 0, size = 5, dropout = 0.5, parent_node = c.layer, name = "f" )

    # Add layers to Model
    model.add_layers( [ c, f ] )

    # Define Specific task Layers
    t0_h = NeuralNode_v1( lr = 0, momentum = 0, size = 3, dropout = 0, parent_node = f.layer, name = "task0_h" )
    t0 = NeuralNode_v1( lr = 0, momentum = 0, size = 2, dropout = 0, parent_node = t0_h.layer, name = "task0", mode = 'categorical', act = tf.nn.softmax )

    t1_h = NeuralNode_v1( lr = 0, momentum = 0, size = 3, dropout = 0, parent_node = f.layer, name = "task1_h" )
    t1 = NeuralNode_v1( lr = 0, momentum = 0, size = 2, dropout = 0, parent_node = t1_h.layer, name = "task1", mode = 'categorical', act = tf.nn.softmax )

    t2_h = NeuralNode_v1( lr = 0, momentum = 0, size = 3, dropout = 0, parent_node = f.layer, name = "task2_h" )
    t2 = NeuralNode_v1( lr = 0, momentum = 0, size = 2, dropout = 0, parent_node = t2_h.layer, name = "task2", mode = 'categorical', act = tf.nn.softmax )

    # Add tasks to Model
    model.add_tasks( [ t0, t1, t2 ] )

    # Initialize model
    model.initilize_model()

    # Train
    batch_size = 10
    step = len( X ) / batch_size

    epoch = 10000000000

    acc = []
    loss = [ ]
    while len(loss) == 0 or loss[ -1 ] > 0.1:

        ac = []
        ls =[]
        for b in range( batch_size ):        
            
            init = int( b * step )
            end = int( init + step ) 

            l, a = model.train( 
                np.asarray( X [ init : end ] ), 
                [ x[ init : end ] for x in Y ] )

            ac.append( a )
            ls.append( l )

        acc.append( np.average( ac, axis=0 ) )
        loss.append( np.average( ls ) )
        
        print("Loss: {}\nAccuracy: {}\n\n".format( np.average( ls ), np.average( ac, axis=0 ) ))