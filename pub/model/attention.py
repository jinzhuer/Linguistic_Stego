import tensorflow as tf
import tensorflow as tf
from .layers.basics import linear, dropout, feed_forward, residual


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.

    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article
    
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    output = inputs * tf.expand_dims(alphas, -1)

    if not return_alphas:
        return output
    else:
        return output, alphas


import tensorflow as tf

from .layers.basics import linear, dropout, feed_forward, residual


def stacked_multihead_attention(x, num_blocks, num_heads, use_residual, is_training, dropout_rate,
                                reuse=False):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attention(x, x, x, use_residual, is_training,
                                                    dropout_rate, num_heads=num_heads, reuse=reuse)
                x = feed_forward(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x, attentions


def multihead_attention(queries, keys, values, use_residual, is_training, dropout_rate,
                        num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attention', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)
        
        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        
        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training, rate=dropout_rate)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)
        
        output = feed_forward(Q_K_V_, num_units, reuse=reuse)
        output = Q_K_V_
        
        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
    return output, attentions


def scaled_dot_product_attention(queries, keys, values, sequence_length=None, reuse=False):
    if sequence_length is None:
        sequence_length = tf.to_float(queries.get_shape().as_list()[-1])
    
    with tf.variable_scope('scaled_dot_product_attention', reuse=reuse):
        keys_T = tf.transpose(keys, [0, 2, 1])
        Q_K = tf.matmul(queries, keys_T) / tf.sqrt(sequence_length)
        attentions = tf.nn.softmax(Q_K)
        scaled_dprod_att = tf.matmul(attentions, values)
    return scaled_dprod_att, attentions



def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing.
    inputs  : 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon : Smoothing rate.
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)

def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding.
    inputs  : 3d tensor. (N, T, E)
    maxlen  : scalar. Must be >= T
    masking : Boolean. If True, padding positions are set to zeros.
    scope   : Optional scope for 'variable_scope'
    Returns :
    3d tensor that has the same shape as inputs.
    '''
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    #with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
                                [pos/np.power(10000, (i-i%2)/E) for i in range(E)]
                                for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2]) # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

    return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr      : initial learning rate. scalar.
    global_step  : scalar.
    warmup_steps : scalar. During warmup_steps, learning rate increases
                   until it reaches init_lr.
    '''
    step = tf.cast(global_step+1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def lr_strategy(init_lr, global_step, type, step_size=200, max_iter=10000):
    step = tf.cast(global_step+1, dtype=tf.float32)
    if type == "poly":
        power = 0.9
        lr = init_lr * (1 - step / max_iter) ** 0.9
    elif type == "sigmoid":
        gamma = 0.05
        lr = init_lr * 1 / (tf.exp(step - step_size) * 0.05)
    return lr