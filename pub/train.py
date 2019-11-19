import tensorflow as tf
import argparse
import os
import pickle as pkl
import numpy as np
from model.word_multi import WordAtt

from data_utils import build_word_dict, build_word_dataset, batch_iter, build_embedding
from data_helper import write_csv_files
from sklearn import metrics
from config_path import get_data_path, get_train_path, log_info


def train(train_x, train_y, valid_x, valid_y, test_x, test_y, vocabulary_size, embed_dict_in, args):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    config.gpu_options.allow_growth = True
    accuracy_file = os.path.join(args.model_dir, 'accuracy.txt')
    with tf.Session(config=config) as sess:
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.num_epochs
        model = WordAtt(vocabulary_size, args.max_document_len, len(args.labels), hidden_layer_num=args.hidden_layers,
                        bi_direction=args.bi_directional, num_hidden=args.num_hidden,
                        embedding_size=args.embedding_size, fc_num_hidden=args.fc_num_hidden,
                        hidden_layer_num_bi=args.hidden_layers_bi, num_hidden_bi=args.num_hidden_bi, embed_dict=embed_dict_in)
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
        def train_step(batch_x, batch_y):
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y,
                model.keep_prob: args.keep_prob  # 0.5
            }
            _, step, loss = sess.run([train_op, global_step, model.loss], feed_dict=feed_dict)
            return step, loss

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

        def train_accuracy():
            _, predictions, ouputs = prediction(train_x, train_y)
            return sum(np.equal(predictions, ouputs)) / len(ouputs)

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
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
        steps = []
        losses = []
        train_acc = []
        test_acc = []
        best_fscore = 0
        last_save_epoch = None
        num_batches_per_epoch = (len(train_y) - 1) // BATCH_SIZE + 1
        for batch_x, batch_y in batches:
            step, loss = train_step(batch_x, batch_y)
            if step % 100 == 0:
                log_info(accuracy_file, "step {0} : loss = {1}".format(step, loss))
            if step % num_batches_per_epoch == 0:  # or (step < num_batches_per_epoch and step % 10 == 0):
                current_epoch = step / num_batches_per_epoch
                if last_save_epoch is not None and current_epoch > 30 and current_epoch - last_save_epoch > 10:
                    break  # early stop
                acc = train_accuracy()
                valid_p, valid_r, valid_f, valid_a, valid_s, valid_aa, _, _ = test_accuracy(valid_x, valid_y)
                if sum(valid_f) / len(valid_f) > best_fscore:
                    last_save_epoch = current_epoch
                    saver.save(sess, os.path.join(args.model_dir, "model.ckpt"), global_step=step)
                    best_fscore = sum(valid_f) / len(valid_f)
                    log_info(accuracy_file, 'new high fscore: %f' % best_fscore)
                write_accuracy(acc, valid_p, valid_r, valid_f, valid_a, valid_s, valid_aa, current_epoch)
                steps.append(step)
                losses.append(loss)
                train_acc.append(acc)
                test_acc.append(valid_aa)
                if loss < 1e-8 or acc > 0.9999:
                    break

        # result
        trained_variables = [v for v in tf.global_variables() if "Adam" not in v.name]
        saver_best = tf.train.Saver(trained_variables)
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        saver_best.restore(sess, ckpt.model_checkpoint_path)
        test_p, test_r, test_f, test_a, test_s, test_aa, labels, predictions = test_accuracy(test_x, test_y)
        log_info(accuracy_file, 'final result on test set:')
        write_accuracy(acc, test_p, test_r, test_f, test_a, test_s, test_aa, last_save_epoch)
        with open(os.path.join(args.model_dir, "LabelsAndPredictions"), "wb") as f:
            final_result = {'labels': labels, 'predictions': predictions}
            pkl.dump(final_result, f)

        def roc_curve(x, y):
            logits, _, outputs = prediction(x, y)
            logits = np.array(logits)
            prob = logits[:, 1] - logits[:, 0]
            return metrics.roc_curve(np.array(outputs), prob, pos_label=1)

        with open(os.path.join(args.model_dir, "LossCurve.pkl"), "wb") as f:
            loss_curve = {'step': steps, 'loss': losses, 'train_acc': train_acc, 'test_acc': test_acc}
            pkl.dump(loss_curve, f)
        fpr, tpr, thresholds = roc_curve(test_x, test_y)
        with open(os.path.join(args.model_dir, "RocCurveData.pkl"), "wb") as f:
            roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
            pkl.dump(roc_data, f)


def logout_config(args, train_y, test_y):
    with open(os.path.join(args.model_dir, "accuracy.txt"), "w") as f:
        print(str(args), file=f)

        labels = list(set(train_y))
        labels.sort()
        print("train samples: %d" % len(train_y), file=f)
        for label in labels:
            print("\t class %d in train set: %d samples" % (label, train_y.count(label)), file=f)

        labels = list(set(test_y))
        labels.sort()
        print("test samples: %d" % len(test_y), file=f)
        for label in labels:
            print("\t class %d in test set: %d samples" % (label, test_y.count(label)), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=str, default="0", help="gpu id to use")
    # parameters for data
    # parser.add_argument("--data_folder", type=str, default="RNN_Huffman", help="ACL | shi")
    parser.add_argument("--data_folder", type=str, default="ACL", help="ACL | shi")
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
    parser.add_argument("--bi_directional", type=str, default="True", help="whether to use bi-directional LSTM")
    parser.add_argument("--fc_num_hidden", type=int, default=256, help="hidden full connect ceil nums before softmax")

    # parameters for training
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="keep prob for drop out")
    parser.add_argument("--up_sample", type=str, default='True', help="whether to up sample data")
    parser.add_argument("--max_document_len", type=int, default=30, help="max length of sentence")
    parser.add_argument("--force_train", type=str, default='True', help='force train exists model')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    args.bi_directional = True if args.bi_directional.lower() in ('yes', 'true', 't', 'y', '1') else False
    args.force_train = True if args.force_train.lower() in ('yes', 'true', 't', 'y', '1') else False
    args.up_sample = True if args.up_sample.lower() in ('yes', 'true', 't', 'y', '1') else False

    train_model = get_train_path(args)
    if os.path.exists(os.path.join(train_model, 'checkpoint')) and not args.force_train:
        print('already trained! %s' % train_model)
    else:
        dataset_dir = get_data_path(args)
        train_text_dirs = []
        test_text_dirs = []
        if not os.path.exists(os.path.join(dataset_dir, 'train')):
            os.makedirs(os.path.join(dataset_dir, 'train'))
        if not os.path.exists(os.path.join(dataset_dir, 'test')):
            os.makedirs(os.path.join(dataset_dir, 'test'))
        for label in args.labels:
            train_text_dir = os.path.join(dataset_dir, 'train',
                                          args.data_type + '_' + str(label) + 'bit_' + str(
                                              args.labeled_data_num) + '.txt')
            test_text_dir = os.path.join(dataset_dir, 'test',
                                         args.data_type + '_' + str(label) + 'bit_' + str(args.test_data_num) + '.txt')
            train_text_dirs.append(train_text_dir)
            test_text_dirs.append(test_text_dir)
            if os.path.exists(train_text_dir) and os.path.exists(test_text_dir):
                continue
            with open(os.path.join(dataset_dir, args.data_type + '_' + str(label) + 'bit.txt'), 'r',
                      encoding='utf8') as f:
                all_lines = f.readlines()

            if args.labeled_data_num < 1:
                train_sample = round(len(all_lines) * args.labeled_data_num)
                test_sample = len(all_lines) - train_sample
            else:
                train_sample = round(args.labeled_data_num)
                test_sample = round(args.test_data_num)
            if not os.path.exists(train_text_dir):
                with open(train_text_dir, 'w', encoding='utf8') as f_train:
                    f_train.writelines(all_lines[:train_sample])
            if not os.path.exists(test_text_dir):
                with open(test_text_dir, 'w', encoding='utf8') as f_test:
                    f_test.writelines(all_lines[-test_sample:])
        model_dir = get_train_path(args)
        args.model_dir = model_dir

        write_csv_files(train_text_dirs, test_text_dirs, args.labels, args.labels, model_dir, 'train.csv', 'test.csv')
        # args.labeled_data_num, args.test_data_num)
        train_path = os.path.join(model_dir, 'train.csv')
        test_path = os.path.join(model_dir, 'test.csv')
        print("\nBuilding dictionary..")
        word_dict = build_word_dict(dataset_dir)
        embed_dict = build_embedding(word_dict,dataset_dir)
        print("Preprocessing dataset..")
        label_map = dict()
        k = 0
        for label in args.labels:
            label_map[label] = k
            k = k + 1
        train_x, train_y, valid_x, valid_y = build_word_dataset(train_path, test_path, "train", word_dict,
                                                                args.max_document_len,
                                                                label_map,
                                                                up_sample=args.up_sample)
        # import pdb
        # pdb.set_trace()
        test_x, test_y = build_word_dataset(train_path, test_path, "test", word_dict, args.max_document_len, label_map,
                                            up_sample=args.up_sample)
        logout_config(args, train_y, test_y)
        train(train_x, train_y, valid_x, valid_y, test_x, test_y, len(word_dict), embed_dict, args)
