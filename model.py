import tensorflow as tf
import model_utils as M

class DLModel(object):
    def __init__(self, batch_size, learning_rate, n_vars,
                 activation=tf.nn.sigmoid, lstm_size = 64, t_window=100):
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.t_window = t_window
        self.n_vars = n_vars
        self.lstm_size = lstm_size
        self.set_graph()
        self.model_saver = tf.train.Saver()
    
    def set_graph(self):
        tf.reset_default_graph()
        self.placeholders = M.NameSpacer(**self.get_placeholders())
        self.forward =  M.NameSpacer(**self.get_forward_operations())
        self.losses =  M.NameSpacer(**self.get_loss())
        self.optimizers = M.NameSpacer(**self.get_optimizer())
        self.metrics = M.NameSpacer(**self.get_metrics())
        self.summaries = M.NameSpacer(**self.get_summaries())
        
    def get_placeholders(self):
        with tf.variable_scope("Placeholders"):
            # train_application
            numeric_input = tf.placeholder(dtype=tf.float32, 
                                           shape=(None, self.t_window, self.n_vars), 
                                           name="ph_numeric_input")
            
            target = tf.placeholder(dtype=tf.float32, 
                                    shape=((None,1)),
                                    name="ph_target")
            
            dropout_1 = tf.placeholder_with_default(1.0, shape=())
            dropout_2 = tf.placeholder_with_default(1.0, shape=())

            loss_weights =  tf.placeholder(dtype=tf.float32, shape=(None, 1))
            
            placeholders = {"numeric_input" : numeric_input,
                            "target" : target,
                            "dropout_1" : dropout_1,
                            "dropout_2" : dropout_2}
            
            return placeholders
    
    def get_forward_operations(self):
        with tf.variable_scope("Forward"):
            with tf.variable_scope("main_flow"):
                input_x  = self.placeholders.numeric_input
                # LSTM definition
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, activation=tf.nn.relu)
                batch_size    = tf.shape(input_x)[1]
                initial_state = lstm_cell.zero_state(batch_size, tf.float32)

                # Passing data through rnn
                rnn_outputs_x, rnn_states_x = tf.nn.dynamic_rnn(lstm_cell, input_x, 
                                                                initial_state=initial_state, 
                                                                time_major=True)
                rnn_outputs_x = rnn_outputs_x[:,-1,:]
                
                dense_output = M.dense_block(rnn_outputs_x, 
                                             num_outputs=32, 
                                             activation=self.activation,
                                             name= "dense_1")
                
                dense_output = M.dense_block(dense_output, 
                                             num_outputs=16, 
                                             activation=self.activation,
                                             name= "dense_2")
                
                dense_output = M.dense_block(dense_output, 
                                             num_outputs=16, 
                                             activation=self.activation,
                                             name= "dense_3")
                
                dense_output = M.dense_block(dense_output, 
                                             num_outputs=8, 
                                             activation=self.activation,
                                             name= "dense_4")
                
                #dense_output_1 = tf.layers.dense(rnn_outputs_x, units=2, activation=self.activation)

                # Output layer
                pred = tf.layers.dense(dense_output, units=1, activation=tf.nn.sigmoid)

            return {"pred" : pred}

    def get_loss(self):
        with tf.variable_scope("Loss"):
            mae = tf.reduce_mean(tf.losses.absolute_difference(labels = self.placeholders.target,
                                                                predictions=self.forward.pred)) 
            
            return {"mae" : mae}
    
    def get_optimizer(self):
        with tf.variable_scope("Optimizer"):
            train_fn = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.losses.mae)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)            
            grads = tf.gradients(self.losses.mae, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, 50)
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars)
            
            return {"optimizer" : train_fn,
                    "optimizer_clip":train_op}

    def get_metrics(self):
        with tf.variable_scope("Accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.abs(self.placeholders.target - self.forward.pred) < 0.5, 
                                              tf.float32))
            return {"accuracy" : accuracy}
        
    def get_summaries(self):
        with tf.variable_scope("Summaries"):
            mae_train = tf.summary.scalar(name="mae_train", tensor=self.losses.mae)
            mae_dev = tf.summary.scalar(name="mae_dev", tensor=self.losses.mae)
            mae_test = tf.summary.scalar(name="mae_test", tensor=self.losses.mae)
            
            
            return({"mae_train": mae_train,
                    "mae_dev": mae_dev,
                    "mae_test": mae_test})