import os
import sys

data_dir = '/data1/yangh/TextStego/TSRNN-Dataset/dataset'
train_model_dir = '/data1/yangh/TextStego/TSRNN-Dataset/train_model'


def get_data_path(args):
    return os.path.join(data_dir, args.data_folder, args.data_type)


def get_train_path(args):
    bi_directional = 'bi_directional' if args.bi_directional else 'uni_directional'
    model_dir = os.path.join(train_model_dir, args.data_folder, args.data_type, str(args.labeled_data_num),
                             bi_directional, '_'.join([str(x) for x in args.labels]))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def log_info(file, text):
    with open(file, 'a') as f:
        print(text, file=f)
    print(text)
    sys.stdout.flush()
