import tensorflow as tf
from tensorflow.contrib import rnn
from .attention import *
import numpy as np
import tensorflow.contrib as tf_contrib

class WordAtt(object):
    def __init__(self, vocabulary_size, max_document_length, num_class, hidden_layer_num=3, embedding_size=256,
                 num_hidden=200, fc_num_hidden=256, bi_direction=False, hidden_layer_num_bi=2, num_hidden_bi=100, embed_dict=0):
        self.embedding_size = embedding_size
        self.bi_direction = bi_direction
        if self.bi_direction:
            self.num_hidden = num_hidden_bi
            self.hidden_layer_num = hidden_layer_num_bi
        else:
            self.num_hidden = num_hidden
            self.hidden_layer_num = hidden_layer_num
        self.fc_num_hidden = fc_num_hidden

        self.x = tf.placeholder(tf.int32, [None, max_document_length])
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.x)[0]
        weigt_decay = 0.0001

        with tf.variable_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            # embeddings = tf.get_variable("embeddings", initializer=embed_dict_in)
            x_emb = tf.nn.embedding_lookup(embeddings, self.x)
            x_pos = positional_encoding(x_emb, max_document_length,masking=True,scope="positional_encoding")
            x_emb = tf.concat([x_emb, x_pos], axis=2)
        with tf.variable_scope("multi"):
            x_att, x_attentions = multihead_attention(x_emb, x_emb, x_emb, use_residual=True, is_training=True,
                                                    dropout_rate=0.9, num_heads=8, reuse=False)
            # x_att, x_attentions = stacked_multihead_attention(x_emb, num_blocks=4, num_heads=8, use_residual=True, is_training=True, dropout_rate=0.8,
                                # reuse=False)
            rnn_output_flat = tf.reduce_mean(x_att, axis=[1])
        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(rnn_output_flat, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)
            self.fc_output = fc_output


        with tf.name_scope("output"):
            self.logits = tf.layers.dense(dropout, num_class)
            # self.logits = tf.layers.dense(dropout, num_class, activation=tf.nn.relu)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            tv = tf.trainable_variables()
            regularization_cost = weigt_decay* tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ]) #0.001是lambda超参数
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)) + regularization_cost

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)