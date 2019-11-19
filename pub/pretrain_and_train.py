import argparse
import os
from config_path import get_train_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=str, default="0", help="gpu id to use")
    # parameters for data
    parser.add_argument("--data_folder", type=str, default="ACL", help="ACL | shi")
    parser.add_argument("--data_type", type=str, default="news",
                        help="for ACL: movie | news | tweet, for shi: 5w_4s | 5w_8s | 7w_4s | 7w_8s")
    parser.add_argument("--unlabeled_data_num", type=int, default=0,
                        help="how many unlabeled data samples was used in pre-train")
    parser.add_argument("--labeled_data_num", type=int, default=8000, help="train data samples for each label")
    parser.add_argument("--test_data_num", type=int, default=2000, help="test data samples for each label")
    parser.add_argument("--labels", nargs='+', type=int, default=[0, 1], help="classes to classify")

    # parameters for model
    parser.add_argument("--bi_directional", type=str, default="False", help="whether to use bi-directional LSTM")
    parser.add_argument("--force_pre_train", type=str, default="False")
    parser.add_argument("--force_train", type=str, default="False")

    args = parser.parse_args()
    args.bi_directional = True if args.bi_directional.lower() in ("yes", "true", "t", "1") else False
    args.force_pre_train = True if args.force_pre_train.lower() in ("yes", "true", "t", "1") else False
    args.force_train = True if args.force_train.lower() in ("yes", "true", "t", "1") else False
    if args.unlabeled_data_num > 0:
        pre_train_model = get_pre_train_path(args)
        if os.path.exists(os.path.join(pre_train_model, 'checkpoint')) and not args.force_pre_train:
            print('already pre-trained! %s' % pre_train_model)
        else:
            os.system(
                'source activate tensorflow && python pre_train.py --GPU=%s --data_folder=%s --data_type=%s --unlabeled_data_num=%d --bi_directional=%s' % (
                    args.GPU, args.data_folder, args.data_type, args.unlabeled_data_num, args.bi_directional))
    train_model = get_train_path(args)
    if os.path.exists(os.path.join(train_model, 'checkpoint')) and not args.force_train:
        print('already trained! %s' % train_model)
    else:
        os.system(
            'source activate tensorflow && python train.py --GPU=%s --data_folder=%s --data_type=%s --unlabeled_data_num=%d --bi_directional=%s --labeled_data_num=%d --test_data_num=%d --labels %s' % (
                args.GPU, args.data_folder, args.data_type, args.unlabeled_data_num, args.bi_directional,
                args.labeled_data_num, args.test_data_num, ' '.join([str(x) for x in args.labels])))
