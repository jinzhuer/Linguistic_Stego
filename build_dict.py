from config_path import get_data_path
from data_utils import *
import argparse


def build_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_size", type=int, default=20000, help="the max size of word dictionary")
    args = parser.parse_args()
    data_folder = 'ACL'
    for data_type in ['movie', 'news', 'tweet']:
        args.data_folder = data_folder
        args.data_type = data_type
        data_path = get_data_path(args)
        word_dict = build_word_dict(data_path, args.dict_size)
        embed_dict = build_embedding(word_dict, data_path)
    # data_folder = 'shi'
    # for data_type in ['5w_4s', '5w_8s', '7w_4s', '7w_8s']:
    #     args.data_folder = data_folder
    #     args.data_type = data_type
    #     data_path = get_data_path(args)
    #     build_word_dict(data_path, args.dict_size)
