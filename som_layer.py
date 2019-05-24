import tensorflow as tf
import numpy as np

class SOMLayer(object):

    def __init__(self, name, m, n, max_epoch, learning_rate_som, radius_factor, act = None):
        
        self.name = name
        self.act = act
        self.m = m
        self.n = n
        self.max_epoch = max_epoch        
        self.alpha = learning_rate_som
        self.sigma = max( m, n ) * radius_factor

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self): return self.smap    
    def getlocation(self): return self.bmu_locs

    def __call__(self, x, is_training=False):

        input_size = x.shape[1].value
        shape =  [ self.m * self.n, input_size ]
        
        self.iter = tf.get_variable( "som_iter_{}".format( self.name ), 1, initializer = tf.constant_initializer(0), trainable = False )
        self.epoch = tf.constant( self.max_epoch, dtype = tf.float32 )
        self.smap = tf.get_variable( "som_w_{}".format( self.name ), shape, initializer = tf.random_normal_initializer( stddev = 0.01 ), trainable = False )
        location_vects = tf.constant( np.array( list( self._neuron_locations( self.m, self.n ) ) ), name = "loc_{}".format( self.name )  )

        grad_pass = tf.pow( tf.subtract( tf.expand_dims( self.smap, axis = 0 ), tf.expand_dims( x, axis = 1 ) ), 2 )        
        squared_distance = tf.reduce_sum( grad_pass, 2 )
        bmu_indices = tf.argmin( squared_distance, axis = 1 )
        self.bmu_locs = tf.reshape( tf.gather( location_vects, bmu_indices ), [ -1, 2 ] )

        updater = None
        updater_iter = None
        if is_training:            
            updater = self.backprop( self.smap, grad_pass, self.bmu_locs, location_vects, x )
            updater_iter =  tf.assign( self.iter[0], tf.cond( self.iter[0] + 1 > self.epoch, lambda: self.iter[0], lambda: self.iter[0] + 1 ) ) 
            
        x = tf.reshape( squared_distance, [ -1, self.m, self.n ] )

        # activation
        if not self.act is None:
            x = self.act( x )

        return tf.expand_dims( x, axis = 3 ), updater, updater_iter

    def backprop(self, smap, grad_pass, bmu_locs, location_vects, x):

        # Update the weigths 
        radius = tf.subtract( self.sigma,
                              tf.multiply( self.iter,
                                           tf.divide( tf.cast( tf.subtract( self.alpha, 1 ), tf.float32 ),
                                                      tf.cast( tf.subtract( self.max_epoch, 1 ),tf.float32 ) ) ) )

        alpha = tf.subtract( self.alpha,
                             tf.multiply( self.iter,
                                          tf.divide( tf.cast( tf.subtract( self.alpha, 1 ), tf.float32 ),
                                                     tf.cast( tf.subtract( self.max_epoch, 1),tf.float32 ) ) ) )

        bmu_distance_squares = tf.reduce_sum(
                tf.pow( tf.subtract(
                    tf.expand_dims( location_vects, axis = 0 ),
                    tf.expand_dims( bmu_locs, axis = 1 ) ), 2 ), 
            2)

        neighbourhood_func = tf.exp( tf.divide( tf.negative( tf.cast(
                bmu_distance_squares, "float32" ) ), tf.multiply(
                tf.square( tf.multiply( radius, 0.08 ) ), 2 ) ) )

        learning_rate_op = tf.multiply( neighbourhood_func, alpha )
        
        numerator = tf.reduce_sum(
            tf.multiply( tf.expand_dims( learning_rate_op, axis =-1 ),
            tf.expand_dims( x, axis = 1 ) ), axis = 0 )

        denominator = tf.expand_dims(
            tf.reduce_sum( learning_rate_op, axis = 0 ) + 1e-8, axis = -1 )

        new_weights = numerator / denominator
        update = tf.assign( smap, new_weights )

        return update
