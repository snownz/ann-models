import sys
sys.path.append('../')

import numpy as np

from ai_utils.models.random_network_destilation import RND

class CuriosityAgent:
    def __init__(
            self,
            n_a,
            n_s,
            sess,
            folder,
            scope,            
            lr=0.01,
            gamma=0.95,
            soft_update=0.95,
            epsilon=1.,
            replace_target_iter=300,
            memory_size=10000,
            batch_size=128,
            s_encode_size=1000,
            h_size=128
    ):
        self.n_a = n_a
        self.n_s = n_s
        self.lr = lr
        self.gamma = gamma
        self.soft_update = soft_update
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = np.zeros( ( self.memory_size, n_s * 2 + 2 ) )
        self.model = RND( s_encode_size, h_size, n_a, folder, scope, sess )

        self.sess = sess

    def build_agent_brain(self):
        self.st, self.ac, self.er, self.s_, self.pred_train, self.dqn_train, self.q, self.soft_update_op, self.dqn_loss = self.model.build_training_graph( self.n_s, self.gamma, self.lr, self.soft_update )

    def store_transition(self, s, a, r, s_): 

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess( self.q, { self.st: s } )
            action = np.argmax( actions_value )
        else:
            action = np.random.randint( 0, self.n_a )
        return action

    def learn(self):

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess( self.soft_update_op )

        # sample batch memory from all memory
        top = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        sample_index = np.random.choice( top, size = self.batch_size )
        batch_memory = self.memory[ sample_index, : ]

        bs, ba, br, bs_ = batch_memory[:, :self.n_s], batch_memory[:, self.n_s], \
            batch_memory[:, self.n_s + 1], batch_memory[:, -self.n_s:]

        _, loss = self.sess( [ self.dqn_train, self.dqn_loss ], { self.st: bs, self.ac: ba, self.er: br, self.s_: bs_ } )
        
        # delay training in order to stay curious
        if self.learn_step_counter % 100 == 0:   
            self.sess( self.pred_train, { self.s_: bs_ } )
        
        self.learn_step_counter += 1

        return loss