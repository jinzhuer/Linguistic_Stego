import tensorflow as tf
import argparse
import os
import pickle as pkl
import numpy as np
from data_utils import build_word_dict, build_word_dataset, batch_iter
from data_helper import write_csv_files
from sklearn import metrics
from config_path import get_data_path, get_train_path, log_info


import tensorflow as tf
from tensorflow.contrib import rnn


class WordRNN(object):
    def __init__(self, vocabulary_size, max_document_length, num_class, hidden_layer_num=3, embedding_size=256,
                 num_hidden=200, fc_num_hidden=256, bi_direction=False, hidden_layer_num_bi=2, num_hidden_bi=100):
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

        with tf.variable_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            x_emb = tf.nn.embedding_lookup(embeddings, self.x)
        with tf.variable_scope("rnn"):
            def lstm_cell():
                return rnn.BasicLSTMCell(self.num_hidden)

            if not self.bi_direction:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(self.hidden_layer_num)])  # , state_is_tuple=True)
                initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
                rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x_emb, initial_state=initial_state, sequence_length=self.x_len,
                                                   dtype=tf.float32)
                # rnn_output_flat = tf.reshape(rnn_outputs, [-1, max_document_length * self.num_hidden])
                rnn_output_flat = tf.reduce_mean(rnn_outputs, axis=1)
            else:
                cell_fw = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(self.hidden_layer_num)])  # , state_is_tuple=True)
                cell_bw = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(self.hidden_layer_num)])  # , state_is_tuple=True)
                initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
                initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_emb,
                                                                 initial_state_fw=initial_state_fw,
                                                                 initial_state_bw=initial_state_bw,
                                                                 sequence_length=self.x_len, dtype=tf.float32)
                # rnn_output_flat = tf.reshape(tf.concat(rnn_outputs, axis=2, name='bidirectional_concat_outputs'),
                #                              [-1, 2 * max_document_length * self.num_hidden])
                rnn_output_flat = tf.reduce_mean(rnn_outputs, axis=[0, 2])
        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(rnn_output_flat, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)
            self.fc_output = fc_output

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(dropout, num_class)
            # self.logits = tf.layers.dense(dropout, num_class, activation=tf.nn.relu)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def test(test_x, test_y, vocabulary_size, args):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    config.gpu_options.allow_growth = True
    accuracy_file = os.path.join(args.model_dir, 'accuracy_final.txt')
    with tf.Session(config=config) as sess:
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.num_epochs
        model = WordRNN(vocabulary_size, args.max_document_len, len(args.labels), hidden_layer_num=args.hidden_layers,
                        bi_direction=args.bi_directional, num_hidden=args.num_hidden,
                        embedding_size=args.embedding_size, fc_num_hidden=args.fc_num_hidden,
                        hidden_layer_num_bi=args.hidden_layers_bi, num_hidden_bi=args.num_hidden_bi)
        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(args.lr)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
        # Checkpoint
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def prediction(x, y):
            batches = batch_iter(x, y, BATCH_SIZE, 1)
            outputs = []
            predictions = []
            logits = []
            for batch_x, batch_y in batches:
                logit, prediction = sess.run([model.logits, model.predictions],
                                             feed_dict={model.x: batch_x, model.y: batch_y,
                                                        model.keep_prob: 1.0})
                logits.extend(logit)
                predictions.extend(prediction.tolist())
                outputs.extend(batch_y.tolist())
            return logits, predictions, outputs

        def test_accuracy(test_x, test_y):
            _, predictions, outputs = prediction(test_x, test_y)
            labels = np.unique(outputs)
            labels_count_TP = np.array([np.sum(b.astype(int)) for b in
                                        [np.logical_and(np.equal(outputs, label_x), np.equal(predictions, label_x)) for
                                         label_x in labels]])
            labels_count_TN = np.array([np.sum(b.astype(int)) for b in [
                np.logical_not(np.logical_or(np.equal(outputs, label_x), np.equal(predictions, label_x))) for label_x in
                labels]])
            labels_count_FP = np.array([np.sum(b.astype(int)) for b in [
                np.logical_and(np.logical_not(np.equal(outputs, label_x)), np.equal(predictions, label_x)) for label_x
                in
                labels]])
            labels_count_FN = np.array([np.sum(b.astype(int)) for b in [
                np.logical_and(np.equal(outputs, label_x), np.logical_not(np.equal(predictions, label_x))) for label_x
                in
                labels]])
            precisions = labels_count_TP / (labels_count_TP + labels_count_FP)
            recalls = labels_count_TP / (labels_count_TP + labels_count_FN)
            fscores = 2 * precisions * recalls / (precisions + recalls)
            accuracies = (labels_count_TP + labels_count_TN) / (
                    labels_count_TP + labels_count_TN + labels_count_FP + labels_count_FN)
            specificities = labels_count_TN / (labels_count_TN + labels_count_FP)
            all_accuracy = np.sum(labels_count_TP) / len(outputs)

            # with open(os.path.join(args.model_dir, "accuracy.txt"), "a") as f:
            #     print("step %d: test_accuracy=%f"%(step,sum_accuracy / cnt), file=f)

            return precisions, recalls, fscores, accuracies, specificities, all_accuracy, outputs, predictions

        def write_accuracy(train_acc, precisions, recalls, fscores, accuracies, specificities, all_accuracy, epoch):
            info = 'epoch %d: train_acc: %f' % (epoch,
                                                train_acc) + '\n' + "epoch %d: precision: %s, recall: %s, fscore: %s, accuracy: %s, specificity: %s, all_accuracy: %s" % (
                       epoch, str(precisions), str(recalls), str(fscores), str(accuracies), str(specificities),
                       str(all_accuracy))
            log_info(accuracy_file, info)

        # Training loop
        # result
        trained_variables = [v for v in tf.global_variables() if "Adam" not in v.name]
        saver_best = tf.train.Saver(trained_variables)
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        saver_best.restore(sess, ckpt.model_checkpoint_path.replace('huffman', args.data_folder))
        test_p, test_r, test_f, test_a, test_s, test_aa, labels, predictions = test_accuracy(test_x, test_y)
        log_info(accuracy_file, 'final result on test set:')
        write_accuracy(0.0, test_p, test_r, test_f, test_a, test_s, test_aa, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=str, default="0", help="gpu id to use")
    # parameters for data
    parser.add_argument("--data_folder", type=str, default="huffman_upsample_mean", help="ACL | shi")
    parser.add_argument("--data_type", type=str, default="news",
                        help="for ACL: movie | news | tweet, for shi: 5w_4s | 5w_8s | 7w_4s | 7w_8s")
    parser.add_argument("--test_data_num", type=float, default=0.2, help="test data samples for each label")
    parser.add_argument("--labels", nargs='+', type=int, default=[0, 1], help="classes to classify")

    # parameters for model
    parser.add_argument("--labeled_data_num", type=float, default=0.8,
                        help="train data samples for each label, smaller than 1 means using all, and the value is the train ratio")
    parser.add_argument("--hidden_layers", type=int, default=3, help="hidden LSTM layer nums")
    parser.add_argument("--embedding_size", type=int, default=256, help="embedding size")
    parser.add_argument("--num_hidden", type=int, default=200, help="hidden LSTM cell nums in each layer")
    parser.add_argument("--hidden_layers_bi", type=int, default=2,
                        help="hidden LSTM layer nums if bi_directional is true")
    parser.add_argument("--num_hidden_bi", type=int, default=100,
                        help="hidden LSTM cell nums in each layer if bi_directional is true")
    parser.add_argument("--bi_directional", type=str, default="False", help="whether to use bi-directional LSTM")
    parser.add_argument("--fc_num_hidden", type=int, default=256, help="hidden full connect ceil nums before softmax")

    # parameters for training
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="keep prob for drop out")
    parser.add_argument("--up_sample", type=str, default='True', help="whether to up sample data")
    parser.add_argument("--max_document_len", type=int, default=30, help="max length of sentence")
    parser.add_argument("--force_train", type=str, default='False', help='force train exists model')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    args.bi_directional = True if args.bi_directional.lower() in ('yes', 'true', 't', 'y', '1') else False
    args.up_sample = True if args.up_sample.lower() in ('yes', 'true', 't', 'y', '1') else False

    train_model = get_train_path(args)
    if not os.path.exists(os.path.join(train_model, 'checkpoint')):
        print('not trained yet! %s' % train_model)
    else:
        dataset_dir = get_data_path(args)
        model_dir = get_train_path(args)
        args.model_dir = model_dir
        # args.labeled_data_num, args.test_data_num)
        train_path = os.path.join(model_dir, 'train.csv')
        test_path = os.path.join(model_dir, 'test.csv')
        print("\nBuilding dictionary..")
        word_dict = build_word_dict(dataset_dir)
        print("Preprocessing dataset..")
        label_map = dict()
        k = 0
        for label in args.labels:
            label_map[label] = k
            k = k + 1
        test_x, test_y = build_word_dataset(train_path, test_path, "test", word_dict, args.max_document_len, label_map,
                                            up_sample=args.up_sample)
        test(test_x, test_y, len(word_dict), args)
