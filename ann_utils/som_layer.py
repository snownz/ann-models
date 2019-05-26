import tensorflow as tf
import numpy as np

class SOMLayer(object):

    def __init__(self, name, m, n, num_epoch, learning_rate_som ,radius_factor, gaussian_std, act=None):
        
        self.m = m
        self.n = n
        self.gaussian_std = gaussian_std
        self.num_epoch = num_epoch       
        self.alpha = learning_rate_som
        self.sigma = max(m,n)*radius_factor
        self.act = act
        self.name = name

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self): return self.map    
    def getlocation(self): return self.bmu_locs

    def __call__(self, x, is_training=False):

        dim = x.shape[-1].value

        self.map = tf.get_variable( "map_{}".format( self.name ), initializer = tf.random_normal( shape = [ self.m * self.n, dim ], stddev = 0.05 ), dtype = tf.float32 )
        self.iter = tf.get_variable( "iter_{}".format( self.name ), initializer = tf.constant( 0.0 ), dtype = tf.float32 )
        
        self.epoch = tf.constant( self.num_epoch, dtype = tf.float32 )         
        self.location_vects = tf.constant( np.array( list( self._neuron_locations( self.m, self.n ) ) ) )    

        self.grad_pass = tf.pow( tf.subtract( tf.expand_dims( self.map, axis = 0 ), tf.expand_dims( x, axis = 1 ) ), 2 )
        self.squared_distance = tf.reduce_sum( self.grad_pass, 2 )
        self.bmu_indices = tf.argmin( self.squared_distance, axis = 1 )
        self.bmu_locs = tf.reshape( tf.gather( self.location_vects, self.bmu_indices ), [-1, 2] )

        updater = None
        if is_training:
            updater = self.backprop( x )
            
        y = tf.reshape( self.squared_distance, [ -1, self.m, self.n ] )

        # activation
        if not self.act is None:
            y = self.act( y )

        return tf.expand_dims( y, axis = 3 ), updater

    def backprop(self, x):

        # Update the weigths 
        self.radius = tf.subtract(self.sigma,
                                tf.multiply(self.iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                    tf.cast(tf.subtract(self.num_epoch, 1),tf.float32))))

        self.alpha = tf.subtract(self.alpha,
                            tf.multiply(self.iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                      tf.cast(tf.subtract(self.num_epoch, 1),tf.float32))))
        self.bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2)

        self.neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(self.radius, self.gaussian_std)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbourhood_func, self.alpha)
        
        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
            tf.expand_dims(x, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op,axis=0) + float(1e-20), axis=-1)

        new_weights = tf.div(self.numerator, self.denominator)
        
        return tf.assign( self.map, new_weights ), \
               tf.assign( self.iter, tf.cond( self.iter < self.epoch, lambda: self.iter + 1.0, lambda: self.epoch ) )
