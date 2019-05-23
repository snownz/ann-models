import tensorflow as tf
import numpy as np

import random, os, collections, gc

from ai_utils.memory_cell_v2 import MemoryCell
from ai_utils.memory_block import MemoryBlock
from ai_utils.helper import flatten
from ai_utils.sess import TfSess
from ai_utils.manager import tf_reset_graph

scope = "test"
mem = {
        "len" : 10,
        "m" : 50,
        "n" : 10,
        "num_epoch" : 10000,
        "lr" : 0.5,
        "nr" : 0.37,
        "act" : tf.nn.tanh,
       }

sess = TfSess( gpu = True )

# Text file containing words for training
training_file = 'rd.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [ word.lower() for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
update_memory_every = 10000
display_step = 100
n_input = 5

# number of units in RNN cell
n_hidden = 64

# tf Graph input
X = tf.placeholder("float", [ None, n_input, 1 ] )
Y = tf.placeholder("float", [ None, vocab_size ] )

mpx = tf.placeholder("float", [ None, 1 ] )
mph = tf.placeholder("float", [ None, n_hidden ] )

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

som = [
        MemoryBlock( "som_{}_mem_lstm".format( i ), mem['m'], mem['n'], mem['num_epoch'], mem['lr'], mem['nr'], mem['act'] )
        for i in range( mem['len'] )
     ]
cell = MemoryCell( n_hidden, "lstm", som )

def RNN(x, weights, biases):

    x = tf.reshape( x, [ -1, n_input ] )

    x = tf.split( x, n_input, 1 )

    # generate prediction
    with tf.variable_scope( scope, reuse = tf.AUTO_REUSE ):

        outputs, states = tf.nn.static_rnn( cell, x, dtype = tf.float32 )

        return [ o[1] for o in outputs ], tf.matmul( outputs[-1][0], weights['out'] ) + biases['out']
 
def update_memory(x, t):

    for m in som:

        # retive memory information
        xi, ti, mi = m( x, t, is_training = True )
        fmi = flatten( mi )
        
        # create input for next layer
        x = tf.concat( [ xi, fmi ], axis = 1 )
        t = tf.concat( [ ti, fmi ], axis = 1 )  

    with tf.control_dependencies( [ x, t ] ):
        return tf.no_op()

def run(offset):

    tf_reset_graph( sess ) 

    sts, pred = RNN(X, weights, biases)

    tf_reset_graph( sess ) 

    sts, pred = RNN(X, weights, biases)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    memory_opt = update_memory( mpx, mph )

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    vars_model = [ x for x in tf.trainable_variables() if scope in x.name ]

    folder = "./model_saved/"

    sess( init )

    saver = tf.train.Saver( vars_model )

    if os.path.isdir( folder ):
        print("Folder: {}".format( folder ) )
        saver.restore( sess.get_session(), "{}{}".format( folder, scope ) )

    step = 0
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    states = []
    xs = []
    while step < training_iters:

        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        try:
            _, acc, loss, onehot_pred, ss = sess([optimizer, accuracy, cost, pred, sts], \
                                                    {X: symbols_in_keys, y: symbols_out_onehot})

            for i, s in enumerate( ss ):
                states.append( np.squeeze( s ) )
                xs.append( symbols_in_keys[0][i] ) 

            if len( states ) > update_memory_every:
                print("Updating Memory")
                o = sess( memory_opt, { mpx : xs, mph: states } )
                states = []
                xs = []    
                                    
            loss_total += loss
            acc_total += acc
            if (step+1) % display_step == 0:
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                    "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                    "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))

                if not os.path.isdir( folder ):
                    os.makedirs( folder )      
                saver.save( sess.get_session(), "{}{}".format( folder, scope ) )

            step += 1
            offset += (n_input+1)
            gc.collect()
        
        except:         
            print( "Error" )
            return offset

    print("Optimization Finished!")
    print("Run on command line.")
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = sess(pred, {X: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:            
            print("Word not in dictionary")
    
    return 0

offset = random.randint(0,n_input+1)
while True:
   offset = run( offset )