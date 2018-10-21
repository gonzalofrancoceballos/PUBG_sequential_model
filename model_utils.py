import tensorflow as tf
import os
import shutil
import tensorflow.contrib.layers as layers
from tensorflow.python.client import device_lib

class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        
def get_tensorflow_configuration(device="0", memory_fraction=1):
    """
    Function for selecting the GPU to use and the amount of memory the process is allowed to use
    :param device: which device should be used (str)
    :param memory_fraction: which proportion of memory must be allocated (float)
    :return: config to be passed to the session (tf object)
    """
    device = str(device)

    if device:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        config.gpu_options.visible_device_list = device
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    return config

def start_tensorflow_session(device="0", memory_fraction=1):
    """
    Starts a tensorflow session taking care of what GPU device is going to be used and
    which is the fraction of memory that is going to be pre-allocated.
    :device: string with the device number (str)
    :memory_fraction: fraction of memory that is going to be pre-allocated in the specified
    device (float [0, 1])
    :return: configured tf.Session
    """
    session_config = get_tensorflow_configuration(device=device, memory_fraction=memory_fraction)
    print(session_config)
    return tf.Session(config=session_config)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def swish(x, name=None):
    '''
    Famous Google activation function
    
    :param x: tf tensor input to the function
    :param name: name of the tensorflow element in the graph 
    :return: tf tensor
    '''
    return tf.multiply(x , tf.nn.sigmoid(x), name=name)


def dense_block(d_input, num_outputs, activation, name, dropout_rate=1):
    '''
    Wrapper function that a simple dense block consisting of 
    dropout--+--layer_norm--+--dense
    
    :param d_input: tf tensor containing the input to the block
    :param num_outputs: since of the dense layer
    :param activation: activation in the dense layer
    :param name: name of the tensorflow element in the graph 
    :param dropout_rate: keep_prob of the dropout layer
    :return: tf operation containing output of the block
    '''
    d_output = tf.nn.dropout(d_input, keep_prob=dropout_rate, name="do_{}.".format(name))
    d_output = LayerNorm(name="do_{}.".format(name))(d_output)
#    d_output = layers.linear(d_output, num_outputs=num_outputs, activation_fn=activation)
    d_output = tf.layers.dense(d_output, units=num_outputs, activation=activation)
    
    return d_output



class LayerNorm(object):
    '''
    Object for layer normalization
    '''
    def __init__(self,  name="layer_norm"):
        '''
        :param name: name of the tensorflow element in the graph 
        '''
        with tf.variable_scope(name):
            self.name = name

    def __call__(self, x, train=True):
        '''
        :param x: tf tensor that contiains the batch that will travel through the layer
        :param train: if true, batch normalization for train stage. If false, 
        batch normalization for predict stage
        :return: layer normalization function
        '''
        return tf.contrib.layers.layer_norm(inputs=x,
                                            center=True,
                                            scale=True,
                                            activation_fn=None,
                                            reuse=None,
                                            variables_collections=None,
                                            outputs_collections=None,
                                            trainable=train,
                                            begin_norm_axis=1,
                                            begin_params_axis=-1,
                                            scope=self.name)

def get_summary_writer(session, logs_path, project_id, version_id):
    """
    For Tensorboard reporting
    :param session: opened tensorflow session (tf.Session)
    :param logs_path: path where tensorboard is looking for logs (str)
    :param project_id: name of the project for reporting purposes (str)
    :param version_id: name of the version for reporting purposes (str)
    :return summary_writer: the tensorboard writer
    """
    path = os.path.join(logs_path,"{}_{}".format(project_id, version_id)) 
    if os.path.exists(path):
        shutil.rmtree(path)
    summary_writer = tf.summary.FileWriter(path, graph_def=session.graph_def)  #graph_def=session.graph
    return(summary_writer)